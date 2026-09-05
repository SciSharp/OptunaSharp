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

    public OptunaStorageProxyService(BaseStorage storage) => _backend = storage;

    /// <summary>The <c>context.abort(code, details=str(e))</c> analog: an aborting gRPC fault.</summary>
    private static RpcException Fault(StatusCode code, Exception e) => new(new Status(code, e.Message));

    // ------------------------------------------------------------------ study manipulation

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

    public override Task<ProtoApi.SetTrialIntermediateValueReply> SetTrialIntermediateValue(
        ProtoApi.SetTrialIntermediateValueRequest request, ServerCallContext context)
    {
        try
        {
            _backend.SetTrialIntermediateValue((int)request.TrialId, (int)request.Step, request.IntermediateValue);
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

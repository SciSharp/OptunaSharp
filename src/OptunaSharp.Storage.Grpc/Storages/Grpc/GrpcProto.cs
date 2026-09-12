using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;                     // ValueError
using OptunaSharp.Collections;      // OrderedDict<,>
using OptunaSharp.Distributions;    // BaseDistribution, Distribution
using OptunaSharp.Internal;         // PyJson
using ProtoApi = global::OptunaSharp.Storages.Grpc.Protobuf;          // the generated protobuf types (proto `package optuna;` → C# ns `Optuna`)

namespace OptunaSharp.Storages.Grpc;

/// <summary>
///     Proto ↔ domain conversion — the C# port of the module-level helpers in
///     <c>optuna/storages/_grpc/servicer.py</c> (<c>_to_proto_trial</c> / <c>_from_proto_trial</c> and
///     the <c>_to_proto_trial_state</c> / <c>_from_proto_trial_state</c> enum mappers), shared by the
///     client (<see cref="GrpcStorageProxy" />) and the server (<see cref="OptunaStorageProxyService" />).
///
///     <para>The wire format is the protobuf contract in <c>Protos/api.proto</c> (vendored verbatim from
///     Optuna). The only encodings that must match Python <i>byte-for-byte inside the proto strings</i>
///     are the ones Optuna computes in Python before handing them to protobuf: distributions via
///     <see cref="Distribution.ToJson" /> (<c>distribution_to_json</c>), attribute values via
///     <see cref="PyJson.Dumps" /> (<c>json.dumps</c>), and the naive-datetime strings in Optuna's
///     <c>DATETIME_FORMAT</c>. Everything else is pure protobuf, identical across languages.</para>
/// </summary>
internal static class GrpcProto
{
    /// <summary>Optuna's <c>DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"</c> (naive, microsecond precision).</summary>
    internal const string DatetimeFormat = "yyyy-MM-dd HH:mm:ss.ffffff";

    // ---------------------------------------------------------------- enum mappers (servicer.py)

    internal static ProtoApi.TrialState ToProtoTrialState(TrialState state) => state switch
    {
        TrialState.Running => ProtoApi.TrialState.Running,
        TrialState.Complete => ProtoApi.TrialState.Complete,
        TrialState.Pruned => ProtoApi.TrialState.Pruned,
        TrialState.Fail => ProtoApi.TrialState.Fail,
        TrialState.Waiting => ProtoApi.TrialState.Waiting,
        _ => throw new ValueError($"Unknown TrialState: {state}"),
    };

    internal static TrialState FromProtoTrialState(ProtoApi.TrialState state) => state switch
    {
        ProtoApi.TrialState.Running => TrialState.Running,
        ProtoApi.TrialState.Complete => TrialState.Complete,
        ProtoApi.TrialState.Pruned => TrialState.Pruned,
        ProtoApi.TrialState.Fail => TrialState.Fail,
        ProtoApi.TrialState.Waiting => TrialState.Waiting,
        _ => throw new ValueError($"Unknown api_pb2.TrialState: {state}"),
    };

    // Optuna maps `StudyDirection.MINIMIZE if d == StudyDirection.MINIMIZE else api_pb2.MAXIMIZE` — a
    // binary choice, so NOT_SET (never persisted on a real study) folds to MAXIMIZE just as it does there.
    internal static ProtoApi.StudyDirection ToProtoDirection(StudyDirection direction)
        => direction == StudyDirection.Minimize ? ProtoApi.StudyDirection.Minimize : ProtoApi.StudyDirection.Maximize;

    internal static StudyDirection FromProtoDirection(ProtoApi.StudyDirection direction)
        => direction == ProtoApi.StudyDirection.Minimize ? StudyDirection.Minimize : StudyDirection.Maximize;

    // ---------------------------------------------------------------- datetime (servicer.py)

    internal static string FormatDatetime(DateTime? value)
        => value is null ? "" : value.Value.ToString(DatetimeFormat, CultureInfo.InvariantCulture);

    internal static DateTime? ParseDatetime(string value)
        => string.IsNullOrEmpty(value)
            ? null
            : DateTime.ParseExact(value, DatetimeFormat, CultureInfo.InvariantCulture);

    // ---------------------------------------------------------------- trial (servicer.py)

    /// <summary>Port of <c>servicer.py: _to_proto_trial</c>.</summary>
    internal static ProtoApi.Trial ToProtoTrial(FrozenTrial trial)
    {
        var proto = new ProtoApi.Trial
        {
            TrialId = trial.TrialId,
            Number = trial.Number,
            State = ToProtoTrialState(trial.State),
            DatetimeStart = FormatDatetime(trial.DatetimeStart),
            DatetimeComplete = FormatDatetime(trial.DatetimeComplete),
        };

        if (trial.Values is not null)
            proto.Values.AddRange(trial.Values);

        foreach (var (key, value) in trial.Params)
            proto.Params[key] = trial.Distributions[key].ToInternalRepr(value);
        foreach (var (key, distribution) in trial.Distributions)
            proto.Distributions[key] = Distribution.ToJson(distribution);
        foreach (var (key, value) in trial.UserAttrs)
            proto.UserAttributes[key] = PyJson.Dumps(value);
        foreach (var (key, value) in trial.SystemAttrs)
            proto.SystemAttributes[key] = PyJson.Dumps(value);
        foreach (var (step, value) in trial.IntermediateValues)
            proto.IntermediateValues[step] = value;

        return proto;
    }

    /// <summary>Port of <c>servicer.py: _from_proto_trial</c>.</summary>
    internal static FrozenTrial FromProtoTrial(ProtoApi.Trial trial)
    {
        var distributions = new OrderedDict<string, BaseDistribution>(trial.Distributions.Count);
        foreach (var (key, value) in trial.Distributions)
            distributions[key] = Distribution.FromJson(value);

        var @params = new OrderedDict<string, object?>(trial.Params.Count);
        foreach (var (key, value) in trial.Params)
            @params[key] = distributions[key].ToExternalRepr(value);

        var userAttrs = new OrderedDict<string, object?>(trial.UserAttributes.Count);
        foreach (var (key, value) in trial.UserAttributes)
            userAttrs[key] = PyJson.Loads(value);

        var systemAttrs = new OrderedDict<string, object?>(trial.SystemAttributes.Count);
        foreach (var (key, value) in trial.SystemAttributes)
            systemAttrs[key] = PyJson.Loads(value);

        var intermediateValues = new OrderedDict<int, double>(trial.IntermediateValues.Count);
        foreach (var (step, value) in trial.IntermediateValues)
            intermediateValues[(int)step] = value;

        return new FrozenTrial(
            number: (int)trial.Number,
            state: FromProtoTrialState(trial.State),
            value: null,
            datetimeStart: ParseDatetime(trial.DatetimeStart),
            datetimeComplete: ParseDatetime(trial.DatetimeComplete),
            @params: @params,
            distributions: distributions,
            userAttrs: userAttrs,
            systemAttrs: systemAttrs,
            intermediateValues: intermediateValues,
            trialId: (int)trial.TrialId,
            values: trial.Values.Count > 0 ? trial.Values.ToList() : null);
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Interop;

/// <summary>How datetimes participate in a snapshot comparison.</summary>
public enum DatetimeComparison
{
    /// <summary>Compare only null-ness (for independently-run studies whose wall clocks differ).</summary>
    NullnessOnly,

    /// <summary>
    ///     Compare as instants at microsecond precision (for states replayed from the same
    ///     journal — naive strings are assumed local, offsets honored, both normalized to UTC).
    /// </summary>
    Instant,
}

/// <summary>One trial's complete observable state.</summary>
public sealed class TrialSnapshot
{
    public long Number;
    public long State;
    public List<double>? Values;
    public OrderedDict<string, object?> Params = new();
    public OrderedDict<string, double> ParamsInternal = new();
    public OrderedDict<string, string> Distributions = new();
    public OrderedDict<string, object?> UserAttrs = new();
    public OrderedDict<string, object?> SystemAttrs = new();
    public OrderedDict<string, double> IntermediateValues = new();
    public string? DatetimeStart;
    public string? DatetimeComplete;
}

/// <summary>
///     A complete, comparison-ready snapshot of a study — the "assert persistence and parity"
///     backbone: capture one from an OptunaSharp <see cref="Study" />, parse one from the
///     Python-side JSON dump, and <see cref="AssertParity" /> them field-by-field with
///     bit-exact float semantics and precise diff messages.
/// </summary>
public sealed class StudySnapshot
{
    public string StudyName = string.Empty;
    public List<long> Directions = new();
    public OrderedDict<string, object?> UserAttrs = new();
    public OrderedDict<string, object?> SystemAttrs = new();
    public List<TrialSnapshot> Trials = new();

    // ------------------------------------------------------------------ capture (C# side)

    public static StudySnapshot Capture(Study study)
    {
        var snapshot = new StudySnapshot
        {
            StudyName = study.StudyName,
            Directions = study.Directions.Select(d => (long)(int)d).ToList(),
            UserAttrs = Normalize(study.UserAttrs),
            SystemAttrs = Normalize(study.Storage.GetStudySystemAttrs(study.StudyId)),
        };

        foreach (var trial in study.GetTrials(deepcopy: true))
        {
            var t = new TrialSnapshot
            {
                Number = trial.Number,
                State = (long)(int)trial.State,
                Values = trial.Values?.ToList(),
                UserAttrs = Normalize(trial.UserAttrsDict),
                SystemAttrs = Normalize(trial.SystemAttrsDict),
                DatetimeStart = FormatDatetime(trial.DatetimeStart),
                DatetimeComplete = FormatDatetime(trial.DatetimeComplete),
            };
            foreach (var (name, value) in trial.ParamsDict)
            {
                t.Params[name] = NormalizeValue(value);
                t.ParamsInternal[name] = trial.DistributionsDict[name].ToInternalRepr(value);
            }
            foreach (var (name, dist) in trial.DistributionsDict)
                t.Distributions[name] = Distribution.ToJson(dist);
            foreach (var (step, value) in trial.IntermediateValues)
                t.IntermediateValues[step.ToString(CultureInfo.InvariantCulture)] = value;
            snapshot.Trials.Add(t);
        }

        return snapshot;
    }

    private static string? FormatDatetime(DateTime? value)
        => value?.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff", CultureInfo.InvariantCulture);

    private static OrderedDict<string, object?> Normalize(IEnumerable<KeyValuePair<string, object?>> attrs)
    {
        var result = new OrderedDict<string, object?>();
        foreach (var (key, value) in attrs)
            result[key] = NormalizeValue(value);
        return result;
    }

    /// <summary>Reduce CLR values to the JSON object model so both sides compare identically.</summary>
    private static object? NormalizeValue(object? value) => value switch
    {
        null => null,
        bool b => b,
        string s => s,
        double d => d,
        float f => (double)f,
        sbyte or byte or short or ushort or int or uint or long =>
            Convert.ToInt64(value, CultureInfo.InvariantCulture),
        CategoricalChoice choice => NormalizeValue(choice.Value),
        OrderedDict<string, object?> dict => Normalize(dict),
        IEnumerable<KeyValuePair<string, object?>> dict => Normalize(dict),
        IEnumerable<object?> list => list.Select(NormalizeValue).ToList(),
        _ => value,
    };

    // ------------------------------------------------------------------ parse (Python side)

    public static StudySnapshot ParsePythonJson(string json)
    {
        var root = (OrderedDict<string, object?>)PyJson.Loads(json)!;
        var snapshot = new StudySnapshot
        {
            StudyName = (string)root["study_name"]!,
            Directions = ((List<object?>)root["directions"]!).Select(Convert.ToInt64).ToList(),
            UserAttrs = (OrderedDict<string, object?>)root["user_attrs"]!,
            SystemAttrs = (OrderedDict<string, object?>)root["system_attrs"]!,
        };
        foreach (var rawTrial in (List<object?>)root["trials"]!)
        {
            var trial = (OrderedDict<string, object?>)rawTrial!;
            var t = new TrialSnapshot
            {
                Number = Convert.ToInt64(trial["number"], CultureInfo.InvariantCulture),
                State = Convert.ToInt64(trial["state"], CultureInfo.InvariantCulture),
                Values = trial["values"] is List<object?> values
                    ? values.Select(v => ToDouble(v)).ToList()
                    : null,
                Params = (OrderedDict<string, object?>)trial["params"]!,
                UserAttrs = (OrderedDict<string, object?>)trial["user_attrs"]!,
                SystemAttrs = (OrderedDict<string, object?>)trial["system_attrs"]!,
                DatetimeStart = (string?)trial["datetime_start"],
                DatetimeComplete = (string?)trial["datetime_complete"],
            };
            foreach (var (k, v) in (OrderedDict<string, object?>)trial["params_internal"]!)
                t.ParamsInternal[k] = ToDouble(v);
            foreach (var (k, v) in (OrderedDict<string, object?>)trial["distributions"]!)
                t.Distributions[k] = (string)v!;
            foreach (var (k, v) in (OrderedDict<string, object?>)trial["intermediate_values"]!)
                t.IntermediateValues[k] = ToDouble(v);
            snapshot.Trials.Add(t);
        }
        return snapshot;
    }

    private static double ToDouble(object? value) => value switch
    {
        double d => d,
        long l => l,
        _ => Convert.ToDouble(value, CultureInfo.InvariantCulture),
    };

    // ------------------------------------------------------------------ parity assertion

    /// <summary>
    ///     Deep, field-by-field parity assertion. Floats compare bit-exact (NaN equals NaN);
    ///     attr trees compare structurally; every mismatch is reported with its path.
    /// </summary>
    public static void AssertParity(
        StudySnapshot expected,
        StudySnapshot actual,
        DatetimeComparison datetimes = DatetimeComparison.NullnessOnly,
        bool compareStudyName = true)
    {
        // Study-state parity is BIT-EXACT (see BitEqual below), so it is host-pinned to x64 — on arm64
        // the .NET libm diverges from NumPy's by ~1 ULP and every FullState/Differential test would go
        // red on a genuine hardware difference. Report Inconclusive there instead (NumSharp's
        // matmul_parity host-pin model). The tolerance-based samplers assert on every architecture.
        PythonSession.EnsureBitExactHostOrInconclusive();

        var diffs = new List<string>();

        if (compareStudyName && expected.StudyName != actual.StudyName)
            diffs.Add($"study_name: {expected.StudyName} != {actual.StudyName}");
        if (!expected.Directions.SequenceEqual(actual.Directions))
            diffs.Add($"directions: [{string.Join(",", expected.Directions)}] != [{string.Join(",", actual.Directions)}]");
        CompareAttrs("study.user_attrs", expected.UserAttrs, actual.UserAttrs, diffs);
        CompareAttrs("study.system_attrs", expected.SystemAttrs, actual.SystemAttrs, diffs);

        if (expected.Trials.Count != actual.Trials.Count)
        {
            diffs.Add($"trial count: {expected.Trials.Count} != {actual.Trials.Count}");
        }
        else
        {
            for (int i = 0; i < expected.Trials.Count; i++)
                CompareTrial(expected.Trials[i], actual.Trials[i], datetimes, diffs);
        }

        if (diffs.Count > 0)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Study state parity FAILED with {diffs.Count} difference(s):");
            foreach (var diff in diffs.Take(40))
                sb.AppendLine("  - " + diff);
            if (diffs.Count > 40)
                sb.AppendLine($"  … and {diffs.Count - 40} more");
            throw new AssertFailedException(sb.ToString());
        }
    }

    private static void CompareTrial(
        TrialSnapshot e, TrialSnapshot a, DatetimeComparison datetimes, List<string> diffs)
    {
        string p = $"trial[{e.Number}]";
        if (e.Number != a.Number)
            diffs.Add($"{p}.number: {e.Number} != {a.Number}");
        if (e.State != a.State)
            diffs.Add($"{p}.state: {e.State} != {a.State}");

        if (e.Values is null != a.Values is null)
        {
            diffs.Add($"{p}.values null-ness: {FormatValues(e.Values)} != {FormatValues(a.Values)}");
        }
        else if (e.Values is not null)
        {
            if (e.Values.Count != a.Values!.Count)
                diffs.Add($"{p}.values length: {e.Values.Count} != {a.Values.Count}");
            else
            {
                for (int i = 0; i < e.Values.Count; i++)
                {
                    if (!BitEqual(e.Values[i], a.Values[i]))
                        diffs.Add($"{p}.values[{i}]: {Repr(e.Values[i])} != {Repr(a.Values[i])}");
                }
            }
        }

        CompareKeyedDoubles($"{p}.params_internal", e.ParamsInternal, a.ParamsInternal, diffs);
        CompareKeyedDoubles($"{p}.intermediate_values", e.IntermediateValues, a.IntermediateValues, diffs);
        CompareAttrs($"{p}.params", e.Params, a.Params, diffs);
        CompareAttrs($"{p}.user_attrs", e.UserAttrs, a.UserAttrs, diffs);
        CompareAttrs($"{p}.system_attrs", e.SystemAttrs, a.SystemAttrs, diffs);

        foreach (var (name, json) in e.Distributions)
        {
            if (!a.Distributions.TryGetValue(name, out var actualJson))
                diffs.Add($"{p}.distributions missing key '{name}'");
            else if (json != actualJson)
                diffs.Add($"{p}.distributions['{name}']: {json} != {actualJson}");
        }
        foreach (var name in a.Distributions.Keys)
        {
            if (!e.Distributions.ContainsKey(name))
                diffs.Add($"{p}.distributions extra key '{name}'");
        }

        CompareDatetime($"{p}.datetime_start", e.DatetimeStart, a.DatetimeStart, datetimes, diffs);
        CompareDatetime($"{p}.datetime_complete", e.DatetimeComplete, a.DatetimeComplete, datetimes, diffs);
    }

    private static void CompareKeyedDoubles(
        string path, OrderedDict<string, double> e, OrderedDict<string, double> a, List<string> diffs)
    {
        foreach (var (key, value) in e)
        {
            if (!a.TryGetValue(key, out double actual))
                diffs.Add($"{path} missing key '{key}'");
            else if (!BitEqual(value, actual))
                diffs.Add($"{path}['{key}']: {Repr(value)} != {Repr(actual)}");
        }
        foreach (var key in a.Keys)
        {
            if (!e.ContainsKey(key))
                diffs.Add($"{path} extra key '{key}'");
        }
    }

    private static void CompareAttrs(
        string path,
        OrderedDict<string, object?> e,
        OrderedDict<string, object?> a,
        List<string> diffs)
    {
        foreach (var (key, value) in e)
        {
            if (!a.TryGetValue(key, out var actual))
                diffs.Add($"{path} missing key '{key}' (expected {PythonRepr.Repr(value)})");
            else
                CompareValues($"{path}['{key}']", value, actual, diffs);
        }
        foreach (var key in a.Keys)
        {
            if (!e.ContainsKey(key))
                diffs.Add($"{path} extra key '{key}' = {PythonRepr.Repr(a[key])}");
        }
    }

    /// <summary>Structural equality over the JSON model: bit-exact doubles, cross int/float numerics equal.</summary>
    private static void CompareValues(string path, object? e, object? a, List<string> diffs)
    {
        switch (e)
        {
            case null:
                if (a is not null)
                    diffs.Add($"{path}: None != {PythonRepr.Repr(a)}");
                return;
            case bool eb:
                if (a is not bool ab || eb != ab)
                    diffs.Add($"{path}: {PythonRepr.Repr(e)} != {PythonRepr.Repr(a)}");
                return;
            case string es:
                if (a is not string @as || es != @as)
                    diffs.Add($"{path}: {PythonRepr.Repr(e)} != {PythonRepr.Repr(a)}");
                return;
            case double ed:
                if (a is double ad)
                {
                    if (!BitEqual(ed, ad))
                        diffs.Add($"{path}: {Repr(ed)} != {Repr(ad)}");
                }
                else if (a is long al)
                {
                    // json round trips may int-ify integral floats on the Python side; accept
                    // numeric equality across the int/float divide (Python == semantics).
                    if (ed != al)
                        diffs.Add($"{path}: {Repr(ed)} != {al}");
                }
                else
                {
                    diffs.Add($"{path}: {Repr(ed)} != {PythonRepr.Repr(a)}");
                }
                return;
            case long el:
                if (a is long al2)
                {
                    if (el != al2)
                        diffs.Add($"{path}: {el} != {al2}");
                }
                else if (a is double ad2)
                {
                    if (el != ad2)
                        diffs.Add($"{path}: {el} != {Repr(ad2)}");
                }
                else
                {
                    diffs.Add($"{path}: {el} != {PythonRepr.Repr(a)}");
                }
                return;
            case List<object?> elist:
                if (a is not List<object?> alist)
                {
                    diffs.Add($"{path}: list != {PythonRepr.Repr(a)}");
                    return;
                }
                if (elist.Count != alist.Count)
                {
                    diffs.Add($"{path}: list length {elist.Count} != {alist.Count}");
                    return;
                }
                for (int i = 0; i < elist.Count; i++)
                    CompareValues($"{path}[{i}]", elist[i], alist[i], diffs);
                return;
            case OrderedDict<string, object?> edict:
                if (a is not OrderedDict<string, object?> adict)
                {
                    diffs.Add($"{path}: dict != {PythonRepr.Repr(a)}");
                    return;
                }
                CompareAttrs(path, edict, adict, diffs);
                return;
            default:
                if (!Equals(e, a))
                    diffs.Add($"{path}: {PythonRepr.Repr(e)} != {PythonRepr.Repr(a)}");
                return;
        }
    }

    private static void CompareDatetime(
        string path, string? e, string? a, DatetimeComparison mode, List<string> diffs)
    {
        if (e is null || a is null)
        {
            if (e is null != a is null)
                diffs.Add($"{path} null-ness: {e ?? "None"} != {a ?? "None"}");
            return;
        }
        if (mode == DatetimeComparison.NullnessOnly)
            return;

        var expectedInstant = ParseInstant(e);
        var actualInstant = ParseInstant(a);
        if (expectedInstant != actualInstant)
            diffs.Add($"{path}: {e} != {a} (as UTC instants: {expectedInstant:O} != {actualInstant:O})");
    }

    private static DateTime ParseInstant(string iso)
        => DateTimeOffset.Parse(iso, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal).UtcDateTime;

    private static bool BitEqual(double e, double a)
        => BitConverter.DoubleToInt64Bits(e) == BitConverter.DoubleToInt64Bits(a)
           || (double.IsNaN(e) && double.IsNaN(a));

    private static string Repr(double v)
        => $"{PythonRepr.FloatRepr(v)} (0x{BitConverter.DoubleToInt64Bits(v):x16})";

    private static string FormatValues(List<double>? values)
        => values is null ? "None" : "[" + string.Join(", ", values.Select(PythonRepr.FloatRepr)) + "]";
}

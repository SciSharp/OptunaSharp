using System.Collections.Generic;
using Microsoft.FSharp.Collections;
using Plotly.NET;
using DObj = DynamicObj.DynamicObj;

namespace OptunaSharp.Visualization.Plotly.Internal;

/// <summary>
///     Raw-C# plotly figure kit. Builds <b>bare</b> <see cref="Trace" />/<see cref="Layout" /> objects
///     (no <c>Chart2D.Chart.*</c> pre-seeding — those inject <c>line:{}</c>/<c>marker:{}</c> defaults
///     Optuna does not emit) and sets properties with the exact plotly keys through
///     <see cref="DObj.SetValue" />. Every trace/layout maps key-for-key to the JSON Optuna's plotly
///     backend produces (<c>go.Scatter</c>/<c>go.Contour</c>/<c>go.Bar</c>/<c>go.Layout</c> …), so the
///     result is structurally diff-able against live Optuna's <c>figure.to_dict()</c>.
///     <para>
///     <b>Null semantics:</b> a <c>null</c> value is <i>omitted</i> — plotly treats an unset property
///     the same as a Python keyword left at its <c>None</c> default (e.g. <c>error_y=None</c> ⇒ the
///     key is simply absent). Empty arrays are <i>not</i> null and are written (<c>x:[]</c>).
///     </para>
/// </summary>
internal static class Fig
{
    /// <summary>A bare trace of the exact plotly type (<c>"scatter"</c>, <c>"contour"</c>, <c>"bar"</c>, <c>"scatter3d"</c>).</summary>
    public static Trace Trace(string type, params (string Key, object? Value)[] entries)
    {
        var t = new Trace(type);
        Fill(t, entries);
        return t;
    }

    /// <summary>A plotly layout (<c>go.Layout</c>) with the given keys.</summary>
    public static Layout Layout(params (string Key, object? Value)[] entries)
    {
        var l = new Layout();
        Fill(l, entries);
        return l;
    }

    /// <summary>A nested plotly object — a <c>marker</c>, <c>error_y</c>, <c>xaxis</c>, <c>colorbar</c>, an annotation, …</summary>
    public static DObj Obj(params (string Key, object? Value)[] entries)
    {
        var o = new DObj();
        Fill(o, entries);
        return o;
    }

    /// <summary>Assemble a figure from bare traces + a layout — <c>go.Figure(data=traces, layout=layout)</c>.</summary>
    public static GenericChart Figure(IEnumerable<Trace> traces, Layout layout)
    {
        var chart = GenericChart.ofTraceObjects(false, ListModule.OfSeq(traces));
        return GenericChart.setLayout(layout, chart);
    }

    /// <summary>Set only the non-null keys (see the null semantics in the type summary).</summary>
    private static void Fill(DObj o, (string Key, object? Value)[] entries)
    {
        foreach (var (key, value) in entries)
            if (value is not null)
                o.SetValue(key, value);
    }
}

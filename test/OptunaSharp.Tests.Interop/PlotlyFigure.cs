using System;
using System.Collections.Generic;
using System.Globalization;
using Newtonsoft.Json.Linq;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Structural comparison of two plotly figure specs — an OptunaSharp Plotly.NET figure
///     (<c>GenericChart.toJson</c> ⇒ <c>{data,layout,config}</c>) against live Optuna's
///     <c>figure.to_dict()</c> (fed in as <c>{data,layout}</c> with the default <c>template</c>
///     stripped, numpy typed-arrays decoded, and scalars flattened — see the Python
///     <c>_oss_fig_json</c> helper). This is exactly Optuna's own visualization-test methodology:
///     assert on <c>figure.data</c> + the meaningful <c>figure.layout</c>, not on pixels.
///     <para>
///     Parsed with Newtonsoft (not System.Text.Json) because plotly serializes non-finite values as
///     the bare JS literals <c>Infinity</c>/<c>-Infinity</c>/<c>NaN</c> (e.g. a running-best over
///     leading infeasible trials is <c>+inf</c>) — Newtonsoft reads those as doubles; STJ rejects them.
///     <b>data</b> is an ordered list of traces, each deep-equal key-for-key; <b>layout</b> is
///     deep-equal after the caller strips plotly.py's auto-injected <c>template</c>. Numbers compare
///     within <paramref name="tol" /> (0 ⇒ bit-exact, as on the pure MT19937 paths; <c>±inf</c> and
///     <c>NaN</c> compare equal to themselves); the plotly <c>title</c> shorthand
///     (<c>"T"</c> ⇔ <c>{"text":"T"}</c>) is normalized on both sides. A key present on one side only
///     is a failure — surfacing a real renderer divergence rather than hiding it.
///     </para>
/// </summary>
internal static class PlotlyFigure
{
    /// <param name="ignoreKeys">
    ///     Object keys to skip anywhere they appear — for values that are inherently non-deterministic
    ///     across two separate runs (e.g. the timeline plot's wall-clock <c>base</c>/<c>x</c>).
    /// </param>
    public static void AssertMatches(string csFigJson, string pyFigJson, double tol = 0.0,
        IReadOnlyCollection<string>? ignoreKeys = null)
    {
        var ignore = ignoreKeys is null ? EmptyIgnore : new HashSet<string>(ignoreKeys);
        var cs = JToken.Parse(csFigJson);
        var py = JToken.Parse(pyFigJson);

        var csData = (JArray)cs["data"]!;
        var pyData = (JArray)py["data"]!;
        if (csData.Count != pyData.Count)
            throw new Exception($"figure.data trace count mismatch: C#={csData.Count}, Python={pyData.Count}");
        for (int i = 0; i < csData.Count; i++)
            AssertNode($"data[{i}]", csData[i], pyData[i], tol, ignore);

        AssertNode("layout", cs["layout"]!, py["layout"]!, tol, ignore);
    }

    private static readonly HashSet<string> EmptyIgnore = new();

    private static void AssertNode(string path, JToken cs, JToken py, double tol, HashSet<string> ignore)
    {
        switch (py.Type)
        {
            case JTokenType.Object:
            {
                if (cs is not JObject cso)
                    throw Mismatch(path, cs, py);
                var pyo = (JObject)py;
                var keys = new HashSet<string>();
                foreach (var p in cso.Properties()) keys.Add(p.Name);
                foreach (var p in pyo.Properties()) keys.Add(p.Name);
                foreach (var k in keys)
                {
                    if (ignore.Contains(k)) continue;
                    JToken? csv = cso[k], pyv = pyo[k];
                    if (csv is null) throw new Exception($"{path}.{k}: present in Python but missing in C# (Python={Trunc(pyv!)})");
                    if (pyv is null) throw new Exception($"{path}.{k}: present in C# but missing in Python (C#={Trunc(csv)})");
                    if (k == "title") { csv = UnwrapTitle(csv); pyv = UnwrapTitle(pyv); }
                    AssertNode($"{path}.{k}", csv, pyv, tol, ignore);
                }
                break;
            }
            case JTokenType.Array:
            {
                if (cs is not JArray csa)
                    throw Mismatch(path, cs, py);
                var pya = (JArray)py;
                if (csa.Count != pya.Count)
                    throw new Exception($"{path}: array length C#={csa.Count}, Python={pya.Count}");
                for (int i = 0; i < csa.Count; i++)
                    AssertNode($"{path}[{i}]", csa[i], pya[i], tol, ignore);
                break;
            }
            case JTokenType.Integer:
            case JTokenType.Float:
            {
                // Plotly.NET's toJson writes non-finites as the strings "Infinity"/"-Infinity"/"NaN";
                // plotly.py's to_dict keeps Python floats (bare Infinity/NaN). Same value, different
                // serializer dialect — coerce both to double before comparing.
                if (!TryGetPlotNumber(cs, out double x))
                    throw Mismatch(path, cs, py);
                double y = py.Value<double>();
                bool ok = x == y
                          || Math.Abs(x - y) <= tol
                          || (double.IsNaN(x) && double.IsNaN(y));
                if (!ok)
                    throw new Exception($"{path}: number C#={x.ToString("R", CultureInfo.InvariantCulture)}, " +
                                        $"Python={y.ToString("R", CultureInfo.InvariantCulture)} (tol={tol})");
                break;
            }
            case JTokenType.String:
            {
                if (cs.Type != JTokenType.String || cs.Value<string>() != py.Value<string>())
                    throw Mismatch(path, cs, py);
                break;
            }
            case JTokenType.Boolean:
            {
                if (cs.Type != JTokenType.Boolean || cs.Value<bool>() != py.Value<bool>())
                    throw Mismatch(path, cs, py);
                break;
            }
            case JTokenType.Null:
            {
                if (cs.Type != JTokenType.Null)
                    throw Mismatch(path, cs, py);
                break;
            }
            default:
                throw new Exception($"{path}: unhandled JSON kind {py.Type}");
        }
    }

    /// <summary>
    ///     A plotly numeric value, tolerating the non-finite string dialect Plotly.NET emits
    ///     (<c>"Infinity"</c>/<c>"-Infinity"</c>/<c>"NaN"</c>) as well as bare floats.
    /// </summary>
    private static bool TryGetPlotNumber(JToken t, out double d)
    {
        if (t.Type is JTokenType.Integer or JTokenType.Float)
        {
            d = t.Value<double>();
            return true;
        }
        if (t.Type == JTokenType.String)
        {
            switch (t.Value<string>())
            {
                case "Infinity": d = double.PositiveInfinity; return true;
                case "-Infinity": d = double.NegativeInfinity; return true;
                case "NaN": d = double.NaN; return true;
            }
        }
        d = 0;
        return false;
    }

    /// <summary>Plotly title shorthand: <c>{"text": X}</c> ⇒ <c>X</c>; a bare scalar is returned as-is.</summary>
    private static JToken UnwrapTitle(JToken e)
        => e is JObject o && o.TryGetValue("text", out var t) ? t : e;

    private static Exception Mismatch(string path, JToken cs, JToken py)
        => new($"{path}: C#({cs.Type})={Trunc(cs)} vs Python({py.Type})={Trunc(py)}");

    private static string Trunc(JToken e)
    {
        string s = e.ToString(Newtonsoft.Json.Formatting.None);
        return s.Length <= 120 ? s : s.Substring(0, 120) + "…";
    }
}

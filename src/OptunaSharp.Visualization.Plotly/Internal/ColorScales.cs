using System;

namespace OptunaSharp.Visualization.Plotly.Internal;

/// <summary>
///     The named plotly colorscales Optuna's renderers pass by value, expanded to the
///     <c>[[position, "rgb(r,g,b)"], …]</c> form plotly's <c>to_dict()</c> emits (evenly-spaced
///     positions, hex → <c>rgb()</c>). Passing the raw hex list would let Plotly.NET expand it its own
///     way; emitting the expanded form keeps the figure spec identical to Optuna's.
/// </summary>
internal static class ColorScales
{
    /// <summary><c>plotly.colors.sequential.Blues</c> — Optuna's <c>COLOR_SCALE</c> (contour, slice, parallel-coordinate).</summary>
    public static readonly object[] Blues = Expand(new[]
    {
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b",
    });

    private static object[] Expand(string[] hex)
    {
        var result = new object[hex.Length];
        for (int i = 0; i < hex.Length; i++)
            result[i] = new object[] { (double)i / (hex.Length - 1), HexToRgb(hex[i]) };
        return result;
    }

    private static string HexToRgb(string hex)
    {
        hex = hex.TrimStart('#');
        int r = Convert.ToInt32(hex.Substring(0, 2), 16);
        int g = Convert.ToInt32(hex.Substring(2, 2), 16);
        int b = Convert.ToInt32(hex.Substring(4, 2), 16);
        return $"rgb({r},{g},{b})";
    }
}

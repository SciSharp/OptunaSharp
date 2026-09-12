using System;
using System.Globalization;
using OptunaSharp.Internal;

namespace OptunaSharp.Visualization.Plotly.Internal;

/// <summary>
///     The <c>plotly.colors</c> helpers Optuna's rank plot needs, ported so the RGB output is
///     byte-identical: <c>sample_colorscale("RdYlBu_r", …)</c> (interpolate on the reversed RdYlBu
///     diverging scale, then <c>convert_to_RGB_255</c> = <c>round(v·255)</c> via a
///     <c>Decimal.ROUND_HALF_EVEN</c> over the <c>repr(float)</c> string). Also the reversed RdYlBu
///     colorscale (for the colorbar trace), <c>np.quantile</c> (linear) for the tick values, and
///     Python <c>%g</c> for the tick labels.
/// </summary>
internal static class PlotlyColors
{
    // plotly.colors.diverging.RdYlBu, reversed (RdYlBu_r), 0-255.
    private static readonly (int R, int G, int B)[] RdYlBuR =
    {
        (49, 54, 149), (69, 117, 180), (116, 173, 209), (171, 217, 233), (224, 243, 248),
        (255, 255, 191), (254, 224, 144), (253, 174, 97), (244, 109, 67), (215, 48, 39), (165, 0, 38),
    };

    /// <summary>
    ///     The reversed-RdYlBu colorscale as a marker's <c>colorscale</c> expands in <c>to_dict()</c>:
    ///     <c>[[i/(n-1), "rgb(r,g,b)"], …]</c>. NB the positions here are <c>i/(n-1)</c> (e.g. exact
    ///     <c>0.3</c>), which differs from the <c>i·step</c> positions <see cref="SampleRgb" />'s
    ///     interpolation uses internally.
    /// </summary>
    /// <returns>Reversed RdYlBu scale expanded into normalized positions and RGB strings.</returns>
    public static object[] RdYlBuRScale()
    {
        int n = RdYlBuR.Length;
        var scale = new object[n];
        for (int i = 0; i < n; i++)
            scale[i] = new object[] { (double)i / (n - 1), $"rgb({RdYlBuR[i].R},{RdYlBuR[i].G},{RdYlBuR[i].B})" };
        return scale;
    }

    /// <summary>
    ///     Python <c>plotly.colors.sample_colorscale("RdYlBu_r", points)</c> → per-point RGB, then the
    ///     <c>label_rgb(unlabel_rgb(...))</c> round-trip Optuna applies, giving <c>"rgb(r.0, g.0, b.0)"</c>.
    /// </summary>
    /// <param name="point">Normalized scale position from zero through one.</param>
    /// <returns>Interpolated RGB text using Plotly-compatible decimal rounding.</returns>
    public static string SampleRgb(double point)
    {
        int n = RdYlBuR.Length;
        double step = 1.0 / (n - 1);

        // scale positions i*step (linspace); bisect_left for the segment.
        int hi = 1;
        while (hi < n - 1 && point >= hi * step)
            hi++;
        int lo = hi - 1;

        double loPos = lo * step, hiPos = hi * step;
        double interp = (point - loPos) / (hiPos - loPos);

        int r = To255(Lerp01(RdYlBuR[lo].R, RdYlBuR[hi].R, interp));
        int g = To255(Lerp01(RdYlBuR[lo].G, RdYlBuR[hi].G, interp));
        int b = To255(Lerp01(RdYlBuR[lo].B, RdYlBuR[hi].B, interp));
        return Rgb(r, g, b);
    }

    /// <summary>Optuna's grey for infeasible trials — <c>(204, 204, 204)</c> in the same float format.</summary>
    public static string InfeasibleRgb => Rgb(204, 204, 204);

    // COLOR_SCALE (plotly.colors.sequential.Blues) as rgb tuples 0-255 — the parallel-coordinate scale.
    private static readonly (int R, int G, int B)[] BluesRgb =
    {
        (247, 251, 255), (222, 235, 247), (198, 219, 239), (158, 202, 225), (107, 174, 214),
        (66, 146, 198), (33, 113, 181), (8, 81, 156), (8, 48, 107),
    };

    /// <summary>
    ///     Optuna's <c>_sample_colorscale(COLOR_SCALE, value)</c> (parallel-coordinate line color):
    ///     <c>find_intermediate_color</c> interpolated in 0-255 rgb space, then <c>label_rgb</c> — so
    ///     the components stay float, e.g. <c>"rgb(20.5, 80.5, 144.0)"</c>.
    /// </summary>
    /// <param name="value">Normalized scale position from zero through one.</param>
    /// <returns>Interpolated Blues color with floating-point RGB components.</returns>
    public static string BluesLineColor(double value)
    {
        int n = BluesRgb.Length;
        double scaled = value * (n - 1);
        int lower = Math.Min((int)scaled, n - 2);
        double t = scaled - lower;
        double r = BluesRgb[lower].R + t * (BluesRgb[lower + 1].R - BluesRgb[lower].R);
        double g = BluesRgb[lower].G + t * (BluesRgb[lower + 1].G - BluesRgb[lower].G);
        double b = BluesRgb[lower].B + t * (BluesRgb[lower + 1].B - BluesRgb[lower].B);
        return $"rgb({PythonRepr.FloatRepr(r)}, {PythonRepr.FloatRepr(g)}, {PythonRepr.FloatRepr(b)})";
    }

    /// <summary>Formats integral color channels with the floating-point notation emitted by Plotly.</summary>
    /// <param name="r">Red channel from zero through 255.</param>
    /// <param name="g">Green channel from zero through 255.</param>
    /// <param name="b">Blue channel from zero through 255.</param>
    /// <returns>An rgb(r.0, g.0, b.0) string.</returns>
    private static string Rgb(int r, int g, int b) => $"rgb({r}.0, {g}.0, {b}.0)";

    // find_intermediate_color interpolates in 0-1 space (colors normalised by /255).
    /// <summary>Interpolates byte-range channels after normalizing both endpoints.</summary>
    /// <param name="lo255">Channel value at the lower color stop.</param>
    /// <param name="hi255">Channel value at the upper color stop.</param>
    /// <param name="interp">Fractional distance between the two stops.</param>
    /// <returns>Interpolated channel in the zero-to-one domain.</returns>
    private static double Lerp01(int lo255, int hi255, double interp)
    {
        double lo = lo255 / 255.0, hi = hi255 / 255.0;
        return lo + interp * (hi - lo);
    }

    // convert_to_RGB_255: Decimal(str(v01*255)).quantize(1, ROUND_HALF_EVEN).
    /// <summary>Converts a normalized channel to an integer using decimal half-even rounding.</summary>
    /// <param name="v01">Normalized finite channel value.</param>
    /// <returns>Rounded channel in the byte domain.</returns>
    /// <exception cref="System.FormatException">The channel cannot be parsed as a decimal.</exception>
    /// <exception cref="System.OverflowException">The scaled channel exceeds Decimal or Int32.</exception>
    private static int To255(double v01)
    {
        decimal d = decimal.Parse(PythonRepr.FloatRepr(v01 * 255.0), CultureInfo.InvariantCulture);
        return (int)Math.Round(d, MidpointRounding.ToEven);
    }

    /// <summary>NumPy <c>np.quantile(sorted, q)</c> with the default linear interpolation.</summary>
    /// <param name="sorted">Nonempty values already sorted in ascending order.</param>
    /// <param name="q">Quantile position from zero through one, inclusive.</param>
    /// <returns>Linearly interpolated quantile matching NumPy's default method.</returns>
    /// <exception cref="System.IndexOutOfRangeException">The input is empty or the quantile produces an index outside the array.</exception>
    public static double QuantileSorted(double[] sorted, double q)
    {
        double pos = q * (sorted.Length - 1);
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        double frac = pos - lo;
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }

    /// <summary>
    ///     Python <c>format(value, "g")</c> (default precision 6): significant-figure formatting with
    ///     trailing zeros stripped, scientific when the exponent is &lt; -4 or ≥ precision.
    /// </summary>
    /// <param name="value">Numeric tick value to format.</param>
    /// <param name="precision">Positive number of significant digits; defaults to six.</param>
    /// <returns>Invariant text with Python-style exponent and trailing-zero conventions.</returns>
    /// <exception cref="System.FormatException">The derived numeric precision is unsupported by the runtime formatter.</exception>
    public static string FormatG(double value, int precision = 6)
    {
        if (value == 0.0)
            return "0";
        if (double.IsNaN(value))
            return "nan";
        if (double.IsInfinity(value))
            return value > 0 ? "inf" : "-inf";

        bool negative = value < 0;
        double abs = Math.Abs(value);
        int exp = (int)Math.Floor(Math.Log10(abs));

        string body;
        if (exp < -4 || exp >= precision)
        {
            // scientific: mantissa with (precision-1) fractional digits, stripped, e±NN (min 2 digits).
            string mantissa = (abs / Math.Pow(10, exp)).ToString("F" + (precision - 1), CultureInfo.InvariantCulture);
            mantissa = StripZeros(mantissa);
            string sign = exp < 0 ? "-" : "+";
            body = mantissa + "e" + sign + Math.Abs(exp).ToString("D2", CultureInfo.InvariantCulture);
        }
        else
        {
            int fracDigits = Math.Max(0, precision - 1 - exp);
            body = StripZeros(abs.ToString("F" + fracDigits, CultureInfo.InvariantCulture));
        }

        return negative ? "-" + body : body;
    }

    /// <summary>Removes fractional trailing zeros and an empty decimal point.</summary>
    /// <param name="s">Fixed-point numeric text to compact.</param>
    /// <returns>The compact numeric text.</returns>
    private static string StripZeros(string s)
    {
        if (!s.Contains('.'))
            return s;
        s = s.TrimEnd('0');
        return s.EndsWith(".") ? s.Substring(0, s.Length - 1) : s;
    }
}

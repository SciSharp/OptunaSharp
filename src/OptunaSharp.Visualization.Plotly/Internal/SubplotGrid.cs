using DObj = DynamicObj.DynamicObj;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Internal;

/// <summary>
///     A faithful port of <c>plotly.subplots.make_subplots(rows, cols, shared_xaxes=True,
///     shared_yaxes=True)</c> for the uniform grids Optuna's contour/rank plots use. Writes the
///     <c>xaxisN</c>/<c>yaxisN</c> layout axes (row-major cell index <c>N = (r-1)·cols + c</c>, default
///     spacing <c>0.2/cols</c> · <c>0.3/rows</c>, per-column/row shared <c>matches</c> +
///     <c>showticklabels:false</c>) and hands back the per-cell axis objects to update
///     (title/range/type) and the <c>x{N}</c>/<c>y{N}</c> anchors to assign traces to.
///     <para>
///     The <c>domain</c> geometry is float-op-order-sensitive plotly boilerplate (not ported
///     visualization logic); render-parity tests exclude the <c>domain</c> key.
///     </para>
/// </summary>
internal sealed class SubplotGrid
{
    private readonly DObj[] _xaxes;   // indexed by N (1-based); [0] unused
    private readonly DObj[] _yaxes;

    /// <summary>Gets the number of subplot rows.</summary>
    public int Rows { get; }
    /// <summary>Gets the number of subplot columns.</summary>
    public int Cols { get; }

    /// <summary>Stores axis objects indexed by one-based subplot number.</summary>
    /// <param name="rows">Positive number of subplot rows.</param>
    /// <param name="cols">Positive number of subplot columns.</param>
    /// <param name="xaxes">X-axis objects with element zero unused.</param>
    /// <param name="yaxes">Y-axis objects with element zero unused.</param>
    private SubplotGrid(int rows, int cols, DObj[] xaxes, DObj[] yaxes)
    {
        Rows = rows;
        Cols = cols;
        _xaxes = xaxes;
        _yaxes = yaxes;
    }

    /// <summary>Cell index N (1-based, row-major) for 1-based (<paramref name="row" />, <paramref name="col" />).</summary>
    /// <param name="row">One-based subplot row.</param>
    /// <param name="col">One-based subplot column.</param>
    /// <returns>One-based row-major cell index; this arithmetic helper does not validate bounds.</returns>
    public int Cell(int row, int col) => (row - 1) * Cols + col;

    /// <summary>Formats the trace-axis reference for a subplot.</summary>
    /// <param name="n">One-based row-major subplot index.</param>
    /// <returns>x or y for the first axis; otherwise the axis prefix followed by its index.</returns>
    public static string XAnchor(int n) => n == 1 ? "x" : $"x{n}";
    /// <summary>Formats the trace-axis reference for a subplot.</summary>
    /// <param name="n">One-based row-major subplot index.</param>
    /// <returns>x or y for the first axis; otherwise the axis prefix followed by its index.</returns>
    public static string YAnchor(int n) => n == 1 ? "y" : $"y{n}";

    /// <summary>Gets the mutable axis object for a subplot.</summary>
    /// <param name="n">One-based row-major subplot index.</param>
    /// <returns>The stored axis object, shared with the chart layout.</returns>
    /// <exception cref="System.IndexOutOfRangeException">The index lies outside the allocated axis array.</exception>
    public DObj XAxis(int n) => _xaxes[n];
    /// <summary>Gets the mutable axis object for a subplot.</summary>
    /// <param name="n">One-based row-major subplot index.</param>
    /// <returns>The stored axis object, shared with the chart layout.</returns>
    /// <exception cref="System.IndexOutOfRangeException">The index lies outside the allocated axis array.</exception>
    public DObj YAxis(int n) => _yaxes[n];

    /// <param name="hSpacing">Horizontal spacing (default plotly <c>0.2/cols</c>; rank passes <c>0.08/cols</c>).</param>
    /// <param name="vSpacing">Vertical spacing (default plotly <c>0.3/rows</c>; rank passes <c>0.08/rows</c>).</param>
    /// <summary>Adds a uniform subplot grid with shared axes to the supplied layout.</summary>
    /// <param name="layout">Plotly layout receiving the axes or attached to the returned chart.</param>
    /// <param name="rows">Positive number of subplot rows.</param>
    /// <param name="cols">Positive number of subplot columns.</param>
    /// <returns>A grid providing the inserted axis objects and their trace references.</returns>
    public static SubplotGrid Init(Layout layout, int rows, int cols, double? hSpacing = null, double? vSpacing = null)
    {
        double hSpace = hSpacing ?? 0.2 / cols;
        double vSpace = vSpacing ?? 0.3 / rows;
        return InitCore(layout, rows, cols, hSpace, vSpace);
    }

    /// <summary>Adds a uniform subplot grid with shared axes to the supplied layout.</summary>
    /// <param name="layout">Plotly layout receiving the axes or attached to the returned chart.</param>
    /// <param name="rows">Positive number of subplot rows.</param>
    /// <param name="cols">Positive number of subplot columns.</param>
    /// <param name="hSpacing">Horizontal gap in normalized paper coordinates.</param>
    /// <param name="vSpacing">Vertical gap in normalized paper coordinates.</param>
    /// <returns>A grid providing the inserted axis objects and their trace references.</returns>
    private static SubplotGrid InitCore(Layout layout, int rows, int cols, double hSpacing, double vSpacing)
    {
        double colW = (1.0 - hSpacing * (cols - 1)) / cols;
        double rowH = (1.0 - vSpacing * (rows - 1)) / rows;

        var xaxes = new DObj[rows * cols + 1];
        var yaxes = new DObj[rows * cols + 1];

        for (int r = 1; r <= rows; r++)
        {
            for (int c = 1; c <= cols; c++)
            {
                int n = (r - 1) * cols + c;

                double x0 = (c - 1) * (colW + hSpacing);
                double yTop = 1.0 - (r - 1) * (rowH + vSpacing);

                var xax = new DObj();
                xax.SetValue("anchor", YAnchor(n));
                xax.SetValue("domain", new[] { x0, x0 + colW });

                var yax = new DObj();
                yax.SetValue("anchor", XAnchor(n));
                yax.SetValue("domain", new[] { yTop - rowH, yTop });

                // shared_xaxes: every row but the bottom matches its column's bottom-row x-axis.
                if (r < rows)
                {
                    xax.SetValue("matches", XAnchor((rows - 1) * cols + c));
                    xax.SetValue("showticklabels", false);
                }

                // shared_yaxes: every column but the leftmost matches its row's leftmost y-axis.
                if (c > 1)
                {
                    yax.SetValue("matches", YAnchor((r - 1) * cols + 1));
                    yax.SetValue("showticklabels", false);
                }

                layout.SetValue(n == 1 ? "xaxis" : $"xaxis{n}", xax);
                layout.SetValue(n == 1 ? "yaxis" : $"yaxis{n}", yax);
                xaxes[n] = xax;
                yaxes[n] = yax;
            }
        }

        return new SubplotGrid(rows, cols, xaxes, yaxes);
    }
}

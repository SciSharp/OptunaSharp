using System;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     AGE-MOEA (2019) survival — port of <c>pymoo.algorithms.moo.age.AGEMOEASurvival</c>
///     (<c>@0.6.1.3</c>), the base that <see cref="AgeMoea2Survival" /> inherits. This partial holds
///     the pieces validated in Phase 2: the extreme-point detector
///     (<see cref="FindCornerSolutions" />), the point-to-line distance it uses, and the hyperplane
///     <see cref="Normalize" /> (whose intercept solve routes through OpenBLAS
///     <c>np.linalg.solve</c>, byte-identical to NumPy). The greedy <c>survival_score</c> /
///     environmental <c>_do</c> and the AGE-MOEA-2 geometry overrides land in Phase 3.
///
///     Panichella, A. (2019), <i>An adaptive evolutionary algorithm based on non-Euclidean geometry
///     for many-objective optimization</i>, GECCO 2019, doi:10.1145/3321707.3321839.
/// </summary>
internal partial class AgeMoeaSurvival
{
    /// <summary>
    ///     <c>point_2_line_distance(P, A, B)</c>: the squared distance from each row of
    ///     <paramref name="p" /> to the line through <paramref name="a" /> and <paramref name="b" />.
    ///     pymoo runs this under <c>@jit(fastmath=True)</c>, but the reductions are over
    ///     <c>n_obj</c> (2–3) terms — no reassociation freedom — so it is bit-exact here.
    /// </summary>
    internal static double[] PointToLineDistance(double[][] p, double[] a, double[] b)
    {
        int m = p.Length;
        int n = a.Length;
        var d = new double[m];

        // ba = B - A;  baba = dot(ba, ba)
        var ba = new double[n];
        double baba = 0.0;
        for (int k = 0; k < n; k++)
        {
            ba[k] = b[k] - a[k];
            baba += ba[k] * ba[k];
        }

        for (int i = 0; i < m; i++)
        {
            double paba = 0.0;
            for (int k = 0; k < n; k++)
                paba += (p[i][k] - a[k]) * ba[k];
            double t = paba / baba;

            double s = 0.0;
            for (int k = 0; k < n; k++)
            {
                double v = (p[i][k] - a[k]) - t * ba[k];
                s += v * v; // (pa - t*ba) ** 2  -> element square
            }
            d[i] = s;
        }
        return d;
    }

    /// <summary>
    ///     <c>find_corner_solutions(front)</c>: the indices of the extreme points — for each axis
    ///     <c>i</c>, the row minimizing the distance to the perturbed axis <c>W[i] = 1e-6 + e_i</c>,
    ///     each row usable once. Returns <c>arange(m)</c> when <c>m ≤ n</c>.
    /// </summary>
    internal static int[] FindCornerSolutions(double[][] front)
    {
        int m = front.Length;
        int n = front[0].Length;

        if (m <= n)
            return Enumerable.Range(0, m).ToArray();

        var zeros = new double[n];
        var indexes = new int[n];
        var selected = new bool[m];

        for (int i = 0; i < n; i++)
        {
            var w = new double[n];
            for (int k = 0; k < n; k++)
                w[k] = 1e-6 + (k == i ? 1.0 : 0.0);

            double[] dists = PointToLineDistance(front, zeros, w);

            int index = -1;
            double best = double.PositiveInfinity;
            for (int r = 0; r < m; r++)
            {
                double dv = selected[r] ? double.PositiveInfinity : dists[r];
                if (index == -1 || dv < best) // np.argmin: first occurrence of the minimum
                {
                    best = dv;
                    index = r;
                }
            }
            indexes[i] = index;
            selected[index] = true;
        }
        return indexes;
    }

    /// <summary>
    ///     <c>normalize(front, extreme)</c>: divide the front by the hyperplane intercepts through the
    ///     extreme points (<c>np.linalg.solve</c>, OpenBLAS byte-exact), falling back to the per-axis
    ///     maximum when the extremes repeat, the solve is singular, or the intercepts are
    ///     non-finite / non-positive. Zero intercepts become 1. Returns the normalized front and the
    ///     normalization vector.
    /// </summary>
    internal static (double[][] Front, double[] Normalization) Normalize(double[][] front, int[] extreme)
    {
        int m = front.Length;
        int n = front[0].Length;

        double[] normalization;

        bool extremesUnique = extreme.Distinct().Count() == extreme.Length;
        if (!extremesUnique)
        {
            normalization = MaxAxis(front, n);
        }
        else
        {
            normalization = SolveIntercepts(front, extreme, n) ?? MaxAxis(front, n);
        }

        for (int j = 0; j < n; j++)
            if (normalization[j] == 0.0)
                normalization[j] = 1.0;

        var outFront = new double[m][];
        for (int i = 0; i < m; i++)
        {
            outFront[i] = new double[n];
            for (int j = 0; j < n; j++)
                outFront[i][j] = front[i][j] / normalization[j];
        }
        return (outFront, normalization);
    }

    /// <summary>The intercepts <c>1 / solve(front[extreme], ones)</c>, or null to trigger the fallback.</summary>
    private static double[]? SolveIntercepts(double[][] front, int[] extreme, int n)
    {
        var a = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[i, j] = front[extreme[i]][j];
        var ones = Enumerable.Repeat(1.0, n).ToArray();

        double[] hyperplane;
        try
        {
            hyperplane = LinAlg.Solve(a, ones);
        }
        catch (Exception) // np.linalg.LinAlgError (singular) -> fall back to per-axis max
        {
            return null;
        }

        if (hyperplane.Any(v => double.IsNaN(v) || double.IsInfinity(v) || v <= 0.0))
            return null;

        var normalization = new double[n];
        for (int j = 0; j < n; j++)
            normalization[j] = 1.0 / hyperplane[j];

        if (normalization.Any(v => double.IsNaN(v) || double.IsInfinity(v)))
            return null;

        return normalization;
    }

    /// <summary><c>np.max(front, axis=0)</c> — the per-objective maximum.</summary>
    private static double[] MaxAxis(double[][] front, int n)
    {
        var mx = new double[n];
        for (int j = 0; j < n; j++)
        {
            double v = double.NegativeInfinity;
            for (int i = 0; i < front.Length; i++)
                if (front[i][j] > v) v = front[i][j];
            mx[j] = v;
        }
        return mx;
    }
}

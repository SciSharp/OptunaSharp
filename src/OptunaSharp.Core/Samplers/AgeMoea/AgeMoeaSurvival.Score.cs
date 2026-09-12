using System;
using System.Collections.Generic;
using System.Linq;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     The AGE-MOEA survival score and environmental selection (continuation of
///     <see cref="AgeMoeaSurvival" />) — port of <c>survival_score</c> / <c>_do</c> /
///     <c>compute_geometry</c> / <c>pairwise_distances</c> / <c>minkowski_distances</c> from
///     <c>pymoo.algorithms.moo.age.AGEMOEASurvival</c> (<c>@0.6.1.3</c>). <see cref="ComputeGeometry" />
///     and <see cref="PairwiseDistances" /> are <c>virtual</c>: <see cref="AgeMoea2Survival" /> overrides
///     them with the AGE-MOEA-2 (2022) geometry.
/// </summary>
internal partial class AgeMoeaSurvival
{
    /// <summary>
    ///     Environmental selection (<c>_do</c>): rank by non-dominated fronts, score front 0 with
    ///     <see cref="SurvivalScore" /> and the later fronts by proximity to the ideal point, then keep
    ///     whole fronts plus the highest-scoring members of the splitting front. Returns the survivors,
    ///     and writes each individual's <see cref="Population.Rank" /> and <see cref="Population.Crowding" />.
    /// </summary>
    /// <param name="pop">Population whose decision and evaluation columns are consumed or updated.</param>
    /// <param name="nSurvive">Maximum number of individuals to retain.</param>
    /// <returns>A population view containing the selected surviving individuals.</returns>
    public Population Do(Population pop, int nSurvive)
        => pop.Subset(SelectMask(pop, nSurvive));

    /// <summary>
    ///     The environmental-selection survivor mask over <paramref name="pop" /> (which is also
    ///     assigned <see cref="Population.Rank" /> and <see cref="Population.Crowding" /> as a side
    ///     effect). Exposed for Layer 2's elite selection, which needs the survivors' ranks/crowding to
    ///     order the parent population; <see cref="Do" /> is the plain subset.
    /// </summary>
    /// <param name="pop">Population whose decision and evaluation columns are consumed or updated.</param>
    /// <param name="nSurvive">Maximum number of individuals to retain.</param>
    /// <returns>A population-row mask identifying survivors.</returns>
    public bool[] SelectMask(Population pop, int nSurvive)
    {
        double[][] f = pop.F!;
        int nInd = f.Length;
        int nObj = f[0].Length;

        var fronts = NonDominatedSorting.Do(f, pop.CV, nSurvive);

        const int maxVal = int.MaxValue;
        var frontNo = Enumerable.Repeat(maxVal, nInd).ToArray();
        for (int i = 0; i < fronts.Count; i++)
            foreach (int idx in fronts[i])
                frontNo[idx] = i;

        pop.Rank = (int[])frontNo.Clone();

        int maxFNo = frontNo.Where(v => v != maxVal).Max();
        var selected = new bool[nInd];
        for (int i = 0; i < nInd; i++)
            selected[i] = frontNo[i] < maxFNo;

        var crowdDist = new double[nInd];

        // front 0 -> survival_score (also yields p and the normalization used by the later fronts)
        int[] front0Idx = Where(frontNo, 0);
        double[][] front1 = front0Idx.Select(i => (double[])f[i].Clone()).ToArray();
        double[] idealPoint = MinAxis(front1, nObj);
        var (cd0, p, normalization) = SurvivalScore(front1, idealPoint);
        for (int k = 0; k < front0Idx.Length; k++)
            crowdDist[front0Idx[k]] = cd0[k];

        // fronts 1 .. max_f_no-1 -> 1 / minkowski_p(front / normalization, ideal)
        for (int fi = 1; fi < maxFNo; fi++)
        {
            int[] idx = Where(frontNo, fi);
            var front = idx.Select(i =>
            {
                var row = new double[nObj];
                for (int j = 0; j < nObj; j++)
                    row[j] = f[i][j] / normalization[j];
                return row;
            }).ToArray();
            double[] md = MinkowskiDistancesToPoint(front, idealPoint, p);
            for (int k = 0; k < idx.Length; k++)
                crowdDist[idx[k]] = 1.0 / md[k];
        }

        // fill the splitting front by descending crowding distance
        int[] last = Where(frontNo, maxFNo);
        int[] order = ArgsortDescending(last.Select(i => crowdDist[i]).ToArray());
        int nAlready = selected.Count(b => b);
        int take = nSurvive - nAlready;
        for (int k = 0; k < take && k < order.Length; k++)
            selected[last[order[k]]] = true;

        pop.Crowding = crowdDist;
        return selected;
    }

    /// <summary>
    ///     <c>survival_score(front, ideal_point)</c> — the greedy maximin diversity score on the first
    ///     front. Rounds to 12 decimals, shifts by the ideal point, normalizes by the extreme-point
    ///     hyperplane, then repeatedly adds the point whose two nearest already-selected neighbours are
    ///     furthest (their summed, norm-scaled distances). Returns the per-row crowding, the geometry
    ///     exponent <c>p</c>, and the normalization vector.
    /// </summary>
    /// <param name="front">Objective vectors of one nondominated front.</param>
    /// <param name="idealPoint">Ideal objective vector used as the normalization origin.</param>
    /// <returns>Diversity scores, the inferred geometry exponent, and objective normalization factors.</returns>
    public (double[] CrowdDist, double P, double[] Normalization) SurvivalScore(double[][] front, double[] idealPoint)
    {
        int m = front.Length;
        int n = front[0].Length;

        // np.round(front, 12) in place
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                front[i][j] = Round12(front[i][j]);

        var crowdDist = new double[m];

        if (m < n)
            return (crowdDist, 1.0, MaxAxis(front, n));

        // shift the ideal point to the origin
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                front[i][j] -= idealPoint[j];

        int[] extreme = FindCornerSolutions(front);
        double[] normalization;
        (front, normalization) = Normalize(front, extreme);

        var selected = new bool[m];
        foreach (int e in extreme)
        {
            crowdDist[e] = double.PositiveInfinity;
            selected[e] = true;
        }

        double p = ComputeGeometry(front, extreme, n);

        // nn = ||front||_p per row;  distances = pairwise / nn[:, None]
        var nn = new double[m];
        for (int i = 0; i < m; i++)
            nn[i] = VectorNorm(front[i], p);

        double[][] sym = PairwiseDistances(front, p);
        // distances[a][b] = sym[a][b] / nn[a]
        var distances = new double[m][];
        for (int a = 0; a < m; a++)
        {
            distances[a] = new double[m];
            for (int b = 0; b < m; b++)
                distances[a][b] = sym[a][b] / nn[a];
        }

        var remaining = new List<int>();
        for (int i = 0; i < m; i++)
            if (!selected[i]) remaining.Add(i);

        int nSelected = selected.Count(b => b);
        for (int iter = 0; iter < m - nSelected; iter++)
        {
            int[] sel = Enumerable.Range(0, m).Where(i => selected[i]).ToArray();

            // D_mg[r][c] = distances[sel[c]][remaining[r]]  (meshgrid(selected, remaining) indexing)
            int bestPos = -1;
            double bestVal = double.NegativeInfinity;
            for (int r = 0; r < remaining.Count; r++)
            {
                double dr;
                if (sel.Length > 1)
                {
                    // sum of the two smallest distances from remaining[r] to the selected set
                    double m1 = double.PositiveInfinity, m2 = double.PositiveInfinity;
                    foreach (int c in sel)
                    {
                        double v = distances[c][remaining[r]];
                        if (v < m1) { m2 = m1; m1 = v; }
                        else if (v < m2) { m2 = v; }
                    }
                    dr = m1 + m2;
                }
                else
                {
                    dr = distances[sel[0]][remaining[r]];
                }

                if (bestPos == -1 || dr > bestVal) // np.argmax: first occurrence of the maximum
                {
                    bestVal = dr;
                    bestPos = r;
                }
            }

            int best = remaining[bestPos];
            remaining.RemoveAt(bestPos);
            selected[best] = true;
            crowdDist[best] = bestVal;
        }

        return (crowdDist, p, normalization);
    }

    /// <summary>
    ///     AGE-MOEA (2019) geometry exponent: <c>p = log(n) / log(1 / mean(front[index]))</c> at the
    ///     diagonal-nearest interior point. Overridden by <see cref="AgeMoea2Survival" /> with the
    ///     <c>find_zero</c> Newton solve.
    /// </summary>
    /// <param name="front">Objective vectors of one nondominated front.</param>
    /// <param name="extreme">Extreme points used to determine objective normalization.</param>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <returns>The estimated p-norm exponent describing the front geometry.</returns>
    internal virtual double ComputeGeometry(double[][] front, int[] extreme, int n)
    {
        double[] d = PointToLineDistance(front, new double[n], Enumerable.Repeat(1.0, n).ToArray());
        foreach (int e in extreme) d[e] = double.PositiveInfinity;
        int index = ArgMin(d);

        double mean = front[index].Sum() / n;
        double p = Math.Log(n) / Math.Log(1.0 / mean);

        if (double.IsNaN(p) || p <= 0.1) p = 1.0;
        else if (p > 20) p = 20.0;
        return p;
    }

    /// <summary>
    ///     AGE-MOEA (2019) pairwise distances: the symmetric Minkowski-<c>p</c> distance matrix.
    ///     Overridden by <see cref="AgeMoea2Survival" /> with the manifold-projected geodesic.
    /// </summary>
    /// <param name="front">Objective vectors of one nondominated front.</param>
    /// <param name="p">Minkowski norm exponent describing the front geometry.</param>
    /// <returns>The pairwise distance matrix in the configured front geometry.</returns>
    internal virtual double[][] PairwiseDistances(double[][] front, double p)
    {
        int m = front.Length;
        var distances = new double[m][];
        for (int i = 0; i < m; i++)
        {
            distances[i] = new double[m];
            for (int j = 0; j < m; j++)
            {
                double s = 0.0;
                for (int k = 0; k < front[i].Length; k++)
                    s += Math.Pow(Math.Abs(front[i][k] - front[j][k]), p);
                distances[i][j] = Math.Pow(s, 1.0 / p);
            }
        }
        return distances;
    }

    /// <summary><c>minkowski_distances(A, ideal[None,:], p)</c> — the Minkowski-<c>p</c> distance of each row to the ideal point.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="point">Objective-space point used for projection or distance measurement.</param>
    /// <param name="p">Minkowski norm exponent describing the front geometry.</param>
    /// <returns>One p-norm distance per input row.</returns>
    internal static double[] MinkowskiDistancesToPoint(double[][] a, double[] point, double p)
    {
        var d = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            double s = 0.0;
            for (int k = 0; k < point.Length; k++)
                s += Math.Pow(Math.Abs(a[i][k] - point[k]), p);
            d[i] = Math.Pow(s, 1.0 / p);
        }
        return d;
    }

    /// <summary>
    ///     <c>np.linalg.norm(x, ord=p)</c> for a vector, mirroring NumPy's branches: <c>ord==1</c> is
    ///     the sum of absolute values (the common case, since the clamp makes <c>p==1.0</c> a mass
    ///     point), <c>ord==2</c> is the Euclidean norm, and the general branch is
    ///     <c>(Σ|x|^p)^(1/p)</c>.
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="p">Minkowski norm exponent describing the front geometry.</param>
    /// <returns>The vector p-norm.</returns>
    internal static double VectorNorm(double[] x, double p)
    {
        if (double.IsPositiveInfinity(p))
            return x.Max(Math.Abs);
        if (p == 1.0)
        {
            double s = 0.0;
            foreach (double v in x) s += Math.Abs(v);
            return s;
        }
        if (p == 2.0)
        {
            double s = 0.0;
            foreach (double v in x) s += v * v;
            return Math.Sqrt(s);
        }
        double acc = 0.0;
        foreach (double v in x) acc += Math.Pow(Math.Abs(v), p);
        return Math.Pow(acc, 1.0 / p);
    }

    /// <summary>
    ///     <c>np.round(x, 12)</c> — round to 12 decimals as NumPy does it: <c>rint(x·1e12)/1e12</c>,
    ///     where <c>rint</c> is round-half-to-even. (This is NOT <c>Math.Round(x, 12)</c>, whose
    ///     decimal-scaling algorithm differs.)
    /// </summary>
    /// <param name="x">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>The value rounded to twelve decimal places using reference-compatible rounding.</returns>
    internal static double Round12(double x)
    {
        const double scale = 1e12;
        return Math.Round(x * scale, MidpointRounding.ToEven) / scale;
    }

    /// <summary>Collects the indices of entries equal to the requested value, preserving input order.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="value">Integer value whose matching positions are selected.</param>
    /// <returns>Matching indices in input order.</returns>
    private static int[] Where(int[] a, int value)
    {
        var idx = new List<int>();
        for (int i = 0; i < a.Length; i++)
            if (a[i] == value) idx.Add(i);
        return idx.ToArray();
    }

    /// <summary>Computes coordinate-wise minima across an objective front.</summary>
    /// <param name="front">Objective vectors of one nondominated front.</param>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <returns>The minimum of each requested objective column.</returns>
    private static double[] MinAxis(double[][] front, int n)
    {
        var mn = new double[n];
        for (int j = 0; j < n; j++)
        {
            double v = double.PositiveInfinity;
            for (int i = 0; i < front.Length; i++)
                if (front[i][j] < v) v = front[i][j];
            mn[j] = v;
        }
        return mn;
    }

    /// <summary>Finds the first index attaining the smallest array value.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The first minimum index.</returns>
    private static int ArgMin(double[] a)
    {
        int idx = 0;
        double best = a[0];
        for (int i = 1; i < a.Length; i++)
            if (a[i] < best) { best = a[i]; idx = i; } // first occurrence
        return idx;
    }

    /// <summary><c>argsort(x)[::-1]</c> — indices ordered by descending value (ascending stable, reversed).</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>Indices ordered from largest to smallest value.</returns>
    private static int[] ArgsortDescending(double[] x)
    {
        // np.argsort ascending (stable on ties by index), then reversed.
        var asc = Enumerable.Range(0, x.Length)
            .OrderBy(i => x[i])
            .ThenBy(i => i)
            .ToArray();
        Array.Reverse(asc);
        return asc;
    }
}

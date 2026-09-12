// JANUS multi-objective CMIX metric functions — a faithful port of the pure-numpy helpers in the
// authors' reference `janus_optuna/janus_moo_cmix.py`: `_standardize`, `_fit_component_metric`,
// `_fit_diag_quadratic_metric`, `_nondominated_indices`, `_crowding_like`.
//
// Reference: JANUS — "Online Jacobian-Aligned Infill for Black-Box Optimization", Hongyuan Yu et al.,
// arXiv:2608.22862 (2026); code https://github.com/hongyuanyu/JANUS (Apache-2.0). The MOO file is
// torch-free (numpy + optuna only), so these fits are byte-comparable against the live reference (the
// same OpenBLAS/LAPACK routes: `@` → LinAlg.Dot, `np.linalg.solve` → column solves) — see
// `JanusMooMetricParityTests`. This is where the MOO sampler's novel geometry lives.
//
// NOTE — the MOO `_standardize` DIFFERS from the single-objective one in `JanusMetric`: it clamps a
// near-zero std to 1.0 (`where(sd < 1e-12, 1, sd)`) rather than adding `+1e-9`, and `_fit_component_metric`
// symmetrizes its output (`0.5*(M+Mᵀ)`). Ported here as a distinct MOO variant, per Plan 05.

using System;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Janus;

/// <summary>
///     The pure-numpy MOO CMIX curvature fits and Pareto helpers of <see cref="JanusMooCmixSampler" />
///     (port of the module-level functions in the reference <c>janus_moo_cmix.py</c>). Given a window of
///     normalized inputs <c>X</c> and either observed components or scalar values, produce a positive
///     semidefinite metric <c>M ≈ JᵀJ</c> whose inverse defines the CMIX exploration covariance.
/// </summary>
internal static class JanusMooMetric
{
    private const double Eps = 1e-12;   // reference _EPS

    /// <summary>
    ///     Reference <c>_standardize</c> (MOO variant): <c>Z = (X - mean) / sd</c> with population std
    ///     (ddof 0), where a std below <see cref="Eps" /> is replaced by 1.0 (not the single-objective
    ///     <c>+1e-9</c>). Returns <c>(Z, sd)</c> — the caller does not need the mean.
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>Standardized observations and the column standard deviations used to scale them.</returns>
    internal static (double[,] Z, double[] Sd) Standardize(double[,] x)
    {
        int n = x.GetLength(0), d = x.GetLength(1);
        var mu = new double[d];
        var sd = new double[d];
        for (int j = 0; j < d; j++) { double s = 0; for (int i = 0; i < n; i++) s += x[i, j]; mu[j] = s / n; }
        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) { double diff = x[i, j] - mu[j]; s += diff * diff; }
            double std = Math.Sqrt(s / n);
            sd[j] = std < Eps ? 1.0 : std;
        }
        var z = new double[n, d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) z[i, j] = (x[i, j] - mu[j]) / sd[j];
        return (z, sd);
    }

    /// <summary>
    ///     Reference <c>_fit_component_metric</c>: from a window of inputs <c>X</c> (n×d) and observed
    ///     component residuals <c>S</c> (n×m), fit the residual Jacobian <c>J = (A⁻¹Zᵀ·Sc)ᵀ/sd</c> and
    ///     return the symmetrized metric <c>M = 0.5(JᵀJ + (JᵀJ)ᵀ)</c>, or <c>null</c> when the window is
    ///     too small / the solve fails / the result is non-finite. Non-finite rows of X or S are dropped.
    ///     <c>Sc = √max(S,0)</c> mean-centered; the ridge solve uses the FULL <c>Zᵀ</c> RHS (n columns),
    ///     matching the reference exactly (byte-comparable).
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="components">Finite component-loss vector read from the trial, when extraction succeeds.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>A regularized component-loss Jacobian Gram matrix, or null when fitting fails.</returns>
    internal static double[,]? FitComponentMetric(double[,] x, double[,] components, double ridge)
    {
        int n0 = x.GetLength(0), d = x.GetLength(1), m = components.GetLength(1);
        if (components.GetLength(0) != n0) return null;
        // finite mask over X rows and S rows
        var (xf, sf, n) = DropNonFiniteRows(x, components);
        if (n < 4 || d <= 0 || m <= 0) return null;

        var (z, sd) = Standardize(xf);
        // Sc = √max(S, 0), then centered per column.
        var sc = new double[n, m];
        for (int i = 0; i < n; i++) for (int c = 0; c < m; c++) sc[i, c] = Math.Sqrt(Math.Max(sf[i, c], 0.0));
        for (int c = 0; c < m; c++)
        {
            double s = 0; for (int i = 0; i < n; i++) s += sc[i, c];
            double mean = s / n;
            for (int i = 0; i < n; i++) sc[i, c] -= mean;
        }

        double[,] gram = LinAlg.Dot(Transpose(z), z);      // ZᵀZ + ridge·I
        for (int i = 0; i < d; i++) gram[i, i] += ridge;
        double[,] zt = Transpose(z);                        // Zᵀ  (d×n)
        double[,]? ainvZt = SolveMatrix(gram, zt);          // A⁻¹ Zᵀ  (d×n)
        if (ainvZt is null) return null;
        double[,] wt = LinAlg.Dot(ainvZt, sc);              // (A⁻¹Zᵀ) Sc  (d×m)
        double[,] wMd = Transpose(wt);                      // W = (…)ᵀ  (m×d)
        var jac = new double[m, d];                         // J = W / sd
        for (int i = 0; i < m; i++) for (int j = 0; j < d; j++) jac[i, j] = wMd[i, j] / sd[j];
        double[,] metric = LinAlg.Dot(Transpose(jac), jac); // JᵀJ  (d×d)
        if (!AllFinite(metric)) return null;
        var sym = new double[d, d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) sym[i, j] = 0.5 * (metric[i, j] + metric[j, i]);
        return sym;
    }

    /// <summary>
    ///     Reference <c>_fit_diag_quadratic_metric</c>: the scalar fallback. Fit a ridge regression of
    ///     <c>y</c> on the design <c>[1 | Z | 0.5·Z⊙Z]</c> (intercept unpenalized) and return
    ///     <c>diag(max(|coef_quad|, ε) / sd²)</c>, or <c>null</c> when the window is too small / the solve
    ///     fails. Non-finite rows of X or entries of y are dropped.
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>A diagonal-quadratic curvature metric, or null when fitting fails.</returns>
    internal static double[,]? FitDiagQuadraticMetric(double[,] x, double[] y, double ridge)
    {
        int d = x.GetLength(1);
        var (xf, yf, n) = DropNonFiniteRows(x, y);
        if (n < 4 || d <= 0) return null;

        var (z, sd) = Standardize(xf);
        int cols = 1 + 2 * d;
        var design = new double[n, cols];
        for (int i = 0; i < n; i++)
        {
            design[i, 0] = 1.0;
            for (int j = 0; j < d; j++) { design[i, 1 + j] = z[i, j]; design[i, 1 + d + j] = 0.5 * z[i, j] * z[i, j]; }
        }
        double[,] gram = LinAlg.Dot(Transpose(design), design);   // designᵀ design
        for (int i = 1; i < cols; i++) gram[i, i] += ridge;       // penalty (intercept unpenalized)
        double[] rhs = LinAlg.Dot(Transpose(design), yf);         // designᵀ y
        double[] coef;
        try { coef = LinAlg.Solve(gram, rhs); }
        catch (Exception) { return null; }
        var hdiag = new double[d];
        for (int j = 0; j < d; j++) hdiag[j] = Math.Max(Math.Abs(coef[1 + d + j]), Eps) / (sd[j] * sd[j]);
        var metric = new double[d, d];
        for (int j = 0; j < d; j++) metric[j, j] = hdiag[j];
        return metric;
    }

    /// <summary>
    ///     Reference <c>_nondominated_indices</c> — ported <b>verbatim</b>. For each still-kept row i it
    ///     clears the rows that <i>dominate</i> i (<c>all(v ≤ v_i) &amp; any(v &lt; v_i)</c>). Non-finite rows
    ///     are excluded up front; if none are finite it returns all indices.
    ///
    ///     <para><b>Faithful-to-a-reference-quirk.</b> This removal order returns the <i>dominated</i> points,
    ///     not the Pareto front — verified against the live reference:
    ///     <c>_nondominated_indices([[0,0],[1,1],[0,1]]) → [1]</c> (the dominated point) and
    ///     <c>([3,1,2]) → [0]</c> (the worst). It is ported exactly as released so the CMIX geometry stays
    ///     byte-comparable to the reference; in practice the center it feeds only shifts a tiny
    ///     (<c>cmix_sigma≈0.002</c>) exploration nudge that NSGA-II selection filters, which is why
    ///     JANUS-MOO <i>matches</i> its host rather than beating it at the conservative default firing rate.</para>
    /// </summary>
    /// <param name="values">Minimization-oriented objective rows used for Pareto ranking and diversity selection.</param>
    /// <returns>Indices of the nondominated minimization-oriented objective rows.</returns>
    internal static int[] NondominatedIndices(double[][] values)
    {
        int n = values.Length;
        var finiteIdx = new System.Collections.Generic.List<int>();
        for (int i = 0; i < n; i++) if (AllFinite(values[i])) finiteIdx.Add(i);
        if (finiteIdx.Count == 0)
        {
            var all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        int k = finiteIdx.Count, m = values[finiteIdx[0]].Length;
        var vals = new double[k][];
        for (int r = 0; r < k; r++) vals[r] = values[finiteIdx[r]];
        var keep = new bool[k];
        for (int r = 0; r < k; r++) keep[r] = true;
        for (int i = 0; i < k; i++)
        {
            if (!keep[i]) continue;
            double[] point = vals[i];
            for (int j = 0; j < k; j++)
            {
                bool allLe = true, anyLt = false;
                for (int c = 0; c < m; c++)
                {
                    if (vals[j][c] > point[c]) { allLe = false; break; }
                    if (vals[j][c] < point[c]) anyLt = true;
                }
                bool dominates = allLe && anyLt;
                if (dominates && j != i) keep[j] = false;
            }
        }
        var outIdx = new System.Collections.Generic.List<int>();
        for (int r = 0; r < k; r++) if (keep[r]) outIdx.Add(finiteIdx[r]);
        return outIdx.ToArray();
    }

    /// <summary>Reference <c>_crowding_like</c>: NSGA-II crowding distance over the rows of
    ///     <paramref name="values" /> (boundary points get <c>+∞</c>). <c>n ≤ 2</c> ⇒ all <c>+∞</c>.</summary>
    /// <param name="values">Minimization-oriented objective rows used for Pareto ranking and diversity selection.</param>
    /// <returns>One normalized objective-space diversity score per row.</returns>
    internal static double[] CrowdingLike(double[][] values)
    {
        int n = values.Length;
        var dist = new double[n];
        if (n <= 2) { for (int i = 0; i < n; i++) dist[i] = double.PositiveInfinity; return dist; }
        int m = values[0].Length;
        for (int j = 0; j < m; j++)
        {
            int[] order = StableArgSort(values, j);
            dist[order[0]] = double.PositiveInfinity;
            dist[order[n - 1]] = double.PositiveInfinity;
            double span = values[order[n - 1]][j] - values[order[0]][j];
            if (!IsFinite(span) || span <= Eps) continue;
            for (int t = 1; t < n - 1; t++)
                dist[order[t]] += (values[order[t + 1]][j] - values[order[t - 1]][j]) / span;
        }
        return dist;
    }

    /// <summary>
    ///     numpy's <c>Generator.multivariate_normal(mean, cov, method="svd")</c> transform for a
    ///     PRE-DRAWN standard-normal vector <paramref name="z" />: factor <c>(U, s, Vh) = svd(cov)</c>
    ///     through the OpenBLAS <c>gesdd</c> NumPy itself calls (byte-exact vs numpy at ANY size, so this
    ///     is not limited to NumSharp's own 4×4 Jacobi <c>multivariate_normal</c>) and return
    ///     <c>mean_j + Σ_i z_i·√s_i·Vh_{ij}</c>. Separating the transform from the PCG64 draw makes it a
    ///     single source of truth shared by <see cref="JanusMooCmix" /> and its byte-parity interop test.
    /// </summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <param name="cov">Initial covariance matrix, or null for the identity default.</param>
    /// <returns>One Gaussian sample transformed with the covariance SVD.</returns>
    internal static double[] MultivariateNormalSvd(double[] z, double[] mean, double[,] cov)
    {
        int d = mean.Length;
        var (_, s, vh) = np.linalg.svd(np.array(cov));   // cov = U·diag(s)·Vh
        double[] sv = s.ToArray<double>();
        double[] vhf = vh.ToArray<double>();             // row-major (d×d)
        var sample = new double[d];
        for (int j = 0; j < d; j++)
        {
            double acc = mean[j];
            for (int i = 0; i < d; i++) acc += z[i] * Math.Sqrt(sv[i]) * vhf[i * d + j];
            sample[j] = acc;
        }
        return sample;
    }

    // ---- helpers ---------------------------------------------------------------------------------

    /// <summary>Rows of (X, S) where all X and all S entries are finite; returns the filtered pair + count.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="s">Numerical state vector.</param>
    /// <returns>Finite input rows, their aligned output rows, and the retained observation count.</returns>
    private static (double[,] X, double[,] S, int N) DropNonFiniteRows(double[,] x, double[,] s)
    {
        int n = x.GetLength(0), d = x.GetLength(1), m = s.GetLength(1);
        var rows = new System.Collections.Generic.List<int>();
        for (int i = 0; i < n; i++)
        {
            bool ok = true;
            for (int j = 0; ok && j < d; j++) ok = IsFinite(x[i, j]);
            for (int c = 0; ok && c < m; c++) ok = IsFinite(s[i, c]);
            if (ok) rows.Add(i);
        }
        var xf = new double[rows.Count, d];
        var sf = new double[rows.Count, m];
        for (int r = 0; r < rows.Count; r++)
        {
            for (int j = 0; j < d; j++) xf[r, j] = x[rows[r], j];
            for (int c = 0; c < m; c++) sf[r, c] = s[rows[r], c];
        }
        return (xf, sf, rows.Count);
    }

    /// <summary>Rows of (X, y) where all X entries and y are finite.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <returns>Finite input rows, their aligned output rows, and the retained observation count.</returns>
    private static (double[,] X, double[] Y, int N) DropNonFiniteRows(double[,] x, double[] y)
    {
        int n = x.GetLength(0), d = x.GetLength(1);
        var rows = new System.Collections.Generic.List<int>();
        for (int i = 0; i < n; i++)
        {
            bool ok = IsFinite(y[i]);
            for (int j = 0; ok && j < d; j++) ok = IsFinite(x[i, j]);
            if (ok) rows.Add(i);
        }
        var xf = new double[rows.Count, d];
        var yf = new double[rows.Count];
        for (int r = 0; r < rows.Count; r++)
        {
            for (int j = 0; j < d; j++) xf[r, j] = x[rows[r], j];
            yf[r] = y[rows[r]];
        }
        return (xf, yf, rows.Count);
    }

    /// <summary>Solve <c>A X = B</c> for a matrix RHS <c>B</c>, column by column (byte-identical to a
    ///     single multi-RHS <c>gesv</c>, since the LU of <c>A</c> is fixed). <c>null</c> on failure / non-finite.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="b">Second numerical array operand.</param>
    /// <returns>The solved coefficient matrix, or null when the numerical solve is unusable.</returns>
    private static double[,]? SolveMatrix(double[,] a, double[,] b)
    {
        int nrhs = b.GetLength(1), dim = a.GetLength(0);
        var result = new double[dim, nrhs];
        try
        {
            for (int c = 0; c < nrhs; c++)
            {
                var col = new double[dim];
                for (int i = 0; i < dim; i++) col[i] = b[i, c];
                double[] sol = LinAlg.Solve(a, col);
                for (int i = 0; i < dim; i++) result[i, c] = sol[i];
            }
        }
        catch (Exception) { return null; }
        return AllFinite(result) ? result : null;
    }

    /// <summary>Allocates a matrix with its row and column axes exchanged.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>A newly allocated transposed matrix.</returns>
    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) t[j, i] = a[i, j];
        return t;
    }

    /// <summary>Indices of <paramref name="values" /> sorted ascending by column <paramref name="col" />,
    ///     ties broken by original index — numpy's stable <c>argsort(kind="mergesort")</c>.</summary>
    /// <param name="values">Minimization-oriented objective rows used for Pareto ranking and diversity selection.</param>
    /// <param name="col">Zero-based column index or column-shaped matrix.</param>
    /// <returns>Row indices ordered by the selected objective, preserving input order for ties.</returns>
    private static int[] StableArgSort(double[][] values, int col)
    {
        int n = values.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) =>
        {
            int cmp = values[a][col].CompareTo(values[b][col]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        return idx;
    }

    /// <summary>Tests whether a scalar excludes NaN and both infinities.</summary>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>True for a finite scalar, including zero.</returns>
    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    /// <summary>Tests whether all entries exclude NaN and both infinities.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>True when every entry is finite.</returns>
    private static bool AllFinite(double[] a) { foreach (double v in a) if (!IsFinite(v)) return false; return true; }

    /// <summary>Tests whether all entries exclude NaN and both infinities.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>True when every entry is finite.</returns>
    private static bool AllFinite(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) if (!IsFinite(a[i, j])) return false;
        return true;
    }
}

// JANUS scalar pseudo-metric machinery — a faithful port of the residual-coordinate fits in the
// authors' reference implementation (`janus_optuna/janus.py`: `pseudo_metric`, `_linear_pseudo`,
// `_fit_full_quadratic`, `_fit_diag_quadratic`, `_fit_pca_quadratic`, `_positive_hessian_metric`,
// `_standardize`, `_pseudo_target`, `resolve_pca_k`).
//
// Reference: JANUS — "Online Jacobian-Aligned Infill for Black-Box Optimization", Hongyuan Yu et al.,
// arXiv:2608.22862 (2026); code https://github.com/hongyuanyu/JANUS (Apache-2.0). JANUS is NOT part
// of Optuna, so — like `LmCmaSampler` — this is a native paper-faithful port validated by convergence,
// with no upstream trajectory to bit-match. numpy ops map to OpenBLAS-backed `LinAlg` (`np.linalg.solve`
// → column solves, `np.linalg.eigh` → `LinAlg.Eigh`, `np.outer` → `LinAlg.Outer`).
//
// These fits turn a window of (x, scalar f) into a local curvature metric G ≈ JᵀJ — the paper's
// "scalar linear/quadratic residual coordinate" mode, used when the objective exposes no component
// residuals (the BBOB setting). `_fit_pca_quadratic`'s SVD basis is obtained via `LinAlg.Eigh(ZᵀZ)`
// (the right-singular vectors of Z are eigenvectors of ZᵀZ); the resulting Hessian is a congruence
// `basis · H_sub · basisᵀ`, invariant to eigenvector sign, so it matches the reference up to the
// convergence tolerance this sampler is validated at.

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Janus;

/// <summary>One local curvature estimate <c>G ≈ JᵀJ</c> produced by <see cref="JanusMetric" />.</summary>
internal sealed class JanusMetricResult
{
    /// <summary>Estimated Jacobian Gram matrix in the original coordinate scale.</summary>
    public required double[,] JtJ { get; init; }
    /// <summary>Local-model feature family used to construct the curvature estimate.</summary>
    public required string Mode { get; init; }
    /// <summary>Estimated numerical rank of the fitted curvature matrix.</summary>
    public int Rank { get; init; }
    /// <summary>Explained variance of the fitted local model.</summary>
    public double FitR2 { get; init; }
    /// <summary>Number of modeled residual or component outputs.</summary>
    public int Q { get; init; } = -1;
}

/// <summary>
///     Scalar residual-coordinate curvature fits (port of the reference's <c>pseudo_metric</c> family).
///     Given a window of inputs <c>X</c> and scalar objective values <c>f</c>, produce a positive
///     semidefinite <c>JtJ</c> that JANUS reuses for the Gauss–Newton step and the CMIX metric.
/// </summary>
internal static class JanusMetric
{
    private const double Eps = 1e-10;

    /// <summary>
    ///     Dispatch on <paramref name="mode" /> exactly as the reference <c>pseudo_metric</c>: quadratic
    ///     modes fall back to <c>linear</c> when the window is too small to fit them, and unknown modes
    ///     default to <c>linear</c>. Returns <c>null</c> when nothing fits (mirrors the numpy <c>None</c>).
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="f">Objective vectors in minimization orientation.</param>
    /// <param name="mode">Feature-family name selecting the scalar local regression model.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <param name="pcaK">Policy selecting the retained principal-component dimension.</param>
    /// <param name="target">Policy for converting scalar objective observations into residual targets.</param>
    /// <returns>The fitted local curvature metric, or null when fitting is disabled or unusable.</returns>
    public static JanusMetricResult? Compute(
        double[,] x, double[] f, string mode = "linear", double ridge = 1.0,
        string pcaK = "auto", string target = "raw")
    {
        int n = x.GetLength(0), d = x.GetLength(1);
        if (n != f.Length || n < 4 || d <= 0)
            return null;
        // Drop non-finite rows (X or f), mirroring the reference's `finite` mask.
        var keep = new bool[n];
        int kept = 0;
        for (int i = 0; i < n; i++)
        {
            bool ok = IsFinite(f[i]);
            for (int j = 0; ok && j < d; j++) ok = IsFinite(x[i, j]);
            keep[i] = ok;
            if (ok) kept++;
        }
        if (kept < 4)
            return null;
        double[,] xc = x;
        double[] fc = f;
        if (kept != n)
        {
            xc = new double[kept, d];
            fc = new double[kept];
            int r = 0;
            for (int i = 0; i < n; i++)
            {
                if (!keep[i]) continue;
                for (int j = 0; j < d; j++) xc[r, j] = x[i, j];
                fc[r] = f[i];
                r++;
            }
        }

        var (z, sd) = Standardize(xc);
        double[] y = PseudoTarget(fc, target);
        mode = (mode ?? "linear").ToLowerInvariant();

        return mode switch
        {
            "none" or "off" or "false" => null,
            "linear" or "lin" => LinearPseudo(z, y, sd, ridge),
            "quadratic" or "quad" => FitFullQuadratic(z, y, sd, ridge) ?? LinearPseudo(z, y, sd, ridge),
            "diag_quadratic" or "diagq" or "diagonal_quadratic"
                => FitDiagQuadratic(z, y, sd, ridge) ?? LinearPseudo(z, y, sd, ridge),
            "pca_quadratic" or "pcaq"
                => FitPcaQuadratic(z, y, sd, pcaK, ridge) ?? LinearPseudo(z, y, sd, ridge),
            "diag_pca_quadratic" or "diag_pcaq" or "diagpca" => DiagPcaQuadratic(z, y, sd, pcaK, ridge),
            _ => LinearPseudo(z, y, sd, ridge),
        };
    }

    /// <summary>Reference <c>resolve_pca_k</c>: the PCA rank for the quadratic fit.</summary>
    /// <param name="pcaK">Policy selecting the retained principal-component dimension.</param>
    /// <param name="n">Number of available observation rows.</param>
    /// <param name="d">Decision-space dimension.</param>
    /// <param name="minExtra">Minimum observation surplus required beyond the feature dimension.</param>
    /// <returns>The number of principal components permitted by the policy and available observations.</returns>
    public static int ResolvePcaK(string pcaK, int n, int d, int minExtra = 10)
    {
        if (pcaK != null && pcaK.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            int budget = Math.Max(1, n - minExtra);
            int qSample = (int)((Math.Sqrt(1.0 + 8.0 * budget) - 3.0) / 2.0);
            int qDim = (int)Math.Ceiling(Math.Sqrt((double)d));
            return Math.Max(1, Math.Min(Math.Min(d, qSample), qDim));
        }
        return int.TryParse(pcaK, out int v) ? Math.Max(1, Math.Min(v, d)) : Math.Max(1, Math.Min(1, d));
    }

    /// <summary>Reference <c>_pseudo_target</c>: shift/rank/log-transform the scalar values, mean-centered.</summary>
    /// <param name="f">Objective vectors in minimization orientation.</param>
    /// <param name="target">Policy for converting scalar objective observations into residual targets.</param>
    /// <returns>Scalar observations converted according to the residual-target policy.</returns>
    public static double[] PseudoTarget(double[] f, string target)
    {
        target = (target ?? "raw").ToLowerInvariant();
        int n = f.Length;
        if (target is "rank" or "ranks" or "rank_normalized")
        {
            int[] order = ArgSort(f);
            var ranks = new double[n];
            for (int r = 0; r < n; r++) ranks[order[r]] = r;
            if (n > 1)
                for (int i = 0; i < n; i++) ranks[i] /= n - 1;
            return Center(ranks);
        }
        if (target is "log" or "log_shift" or "log1p")
        {
            double shift = Min(f);
            var yv = new double[n];
            for (int i = 0; i < n; i++) yv[i] = Math.Log(1.0 + Math.Max(f[i] - shift, 0.0)); // log1p
            return Center(yv);
        }
        return Center(f);
    }

    // ---- standardization -------------------------------------------------------------------------

    /// <summary>Reference <c>_standardize</c>: <c>Z = (X - mean) / (std + 1e-9)</c>, population std (ddof 0).</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>Standardized observations and the column standard deviations used to scale them.</returns>
    private static (double[,] Z, double[] Sd) Standardize(double[,] x)
    {
        int n = x.GetLength(0), d = x.GetLength(1);
        var mu = new double[d];
        var sd = new double[d];
        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) s += x[i, j];
            mu[j] = s / n;
        }
        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) { double diff = x[i, j] - mu[j]; s += diff * diff; }
            sd[j] = Math.Sqrt(s / n) + 1e-9;
        }
        var z = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                z[i, j] = (x[i, j] - mu[j]) / sd[j];
        return (z, sd);
    }

    // ---- linear ----------------------------------------------------------------------------------

    /// <summary>Reference <c>_linear_pseudo</c>: slope <c>g = solve(ZᵀZ+ρI, Zᵀy)/sd</c>, <c>JtJ = outer(g,g)</c>.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="sd">Per-coordinate scales used to standardize the numerical observations.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>A scalar linear residual metric, or null if the regression is unusable.</returns>
    private static JanusMetricResult? LinearPseudo(double[,] z, double[] y, double[] sd, double ridge)
    {
        int d = z.GetLength(1);
        double[]? w = RidgeSolve(z, y, ridge);
        if (w is null) return null;
        var g = new double[d];
        for (int i = 0; i < d; i++) g[i] = w[i] / sd[i];
        if (!AllFinite(g)) return null;
        return new JanusMetricResult { JtJ = LinAlg.Outer(g, g), Mode = "linear", Rank = 1, FitR2 = 0.0 };
    }

    // ---- full quadratic --------------------------------------------------------------------------

    /// <summary>Fits all second-order feature interactions to estimate local scalar-objective curvature.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="sd">Per-coordinate scales used to standardize the numerical observations.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>The fitted metric, or null when the local regression is not usable.</returns>
    private static JanusMetricResult? FitFullQuadratic(double[,] z, double[] y, double[] sd, double ridge)
    {
        int n = z.GetLength(0), d = z.GetLength(1);
        var (iu0, iu1) = TriuIndices(d);
        int pairs = iu0.Length;
        int cols = d + pairs;
        if (n < cols + 2) return null;
        // Phi = [Z | Z[:,iu0]*Z[:,iu1]]
        var phi = new double[n, cols];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++) phi[i, j] = z[i, j];
            for (int p = 0; p < pairs; p++) phi[i, d + p] = z[i, iu0[p]] * z[i, iu1[p]];
        }
        double[]? coef = RidgeSolve(phi, y, ridge);
        if (coef is null) return null;
        double fitR2 = FitR2(phi, coef, y);
        // H_std[iu] = coef[d:]; H_std += H_stdᵀ  (diagonal doubled — the true Hessian of coef·z²).
        var hStd = new double[d, d];
        for (int p = 0; p < pairs; p++) hStd[iu0[p], iu1[p]] = coef[d + p];
        var h = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                h[i, j] = (hStd[i, j] + hStd[j, i]) / (sd[i] * sd[j]);
        return PositiveHessianMetric(h, "quadratic", fitR2);
    }

    // ---- diagonal quadratic ----------------------------------------------------------------------

    /// <summary>Fits coordinate-wise quadratic features to estimate diagonal local curvature.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="sd">Per-coordinate scales used to standardize the numerical observations.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>The fitted metric, or null when the local regression is not usable.</returns>
    private static JanusMetricResult? FitDiagQuadratic(double[,] z, double[] y, double[] sd, double ridge)
    {
        int n = z.GetLength(0), d = z.GetLength(1);
        if (n < 2 * d + 10) return null;
        var phi = new double[n, 2 * d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++) { phi[i, j] = z[i, j]; phi[i, d + j] = z[i, j] * z[i, j]; }
        }
        double[]? coef = RidgeSolve(phi, y, ridge);
        if (coef is null) return null;
        double fitR2 = FitR2(phi, coef, y);
        var hdiag = new double[d];
        int nz = 0;
        for (int j = 0; j < d; j++)
        {
            hdiag[j] = Math.Max(2.0 * coef[d + j] / (sd[j] * sd[j]), 0.0);
            if (hdiag[j] > 0) nz++;
        }
        if (nz == 0) return null;
        return new JanusMetricResult { JtJ = Diag(hdiag), Mode = "diag_quadratic", Rank = nz, FitR2 = fitR2 };
    }

    // ---- PCA quadratic (SVD basis via eigh(ZᵀZ)) -------------------------------------------------

    /// <summary>Fits a quadratic model in principal-component coordinates to reduce the feature dimension.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="sd">Per-coordinate scales used to standardize the numerical observations.</param>
    /// <param name="pcaK">Policy selecting the retained principal-component dimension.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>The fitted metric, or null when the local regression is not usable.</returns>
    private static JanusMetricResult? FitPcaQuadratic(double[,] z, double[] y, double[] sd, string pcaK, double ridge)
    {
        int n = z.GetLength(0), d = z.GetLength(1);
        if (pcaK != null && (pcaK.Equals("none", StringComparison.OrdinalIgnoreCase)
            || pcaK.Equals("off", StringComparison.OrdinalIgnoreCase)
            || pcaK.Equals("false", StringComparison.OrdinalIgnoreCase) || pcaK == "0"))
            return null;
        int q = Math.Max(1, Math.Min(ResolvePcaK(pcaK, n, d, 10), n - 2));
        double[,]? basis = TopRightSingularVectors(z, q);   // d×q, columns = top-q directions
        if (basis is null) return null;
        int qEff = basis.GetLength(1);
        // U = Z @ basis  (n×q)
        double[,] u = LinAlg.Dot(z, basis);
        var (iu0, iu1) = TriuIndices(qEff);
        int pairs = iu0.Length;
        int cols = qEff + pairs;
        if (n < cols + 2) return null;
        var phi = new double[n, cols];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < qEff; j++) phi[i, j] = u[i, j];
            for (int p = 0; p < pairs; p++) phi[i, qEff + p] = u[i, iu0[p]] * u[i, iu1[p]];
        }
        double[]? coef = RidgeSolve(phi, y, ridge);
        if (coef is null) return null;
        double fitR2 = FitR2(phi, coef, y);
        var hSub = new double[qEff, qEff];
        for (int p = 0; p < pairs; p++) hSub[iu0[p], iu1[p]] = coef[qEff + p];
        var hSubSym = new double[qEff, qEff];
        for (int i = 0; i < qEff; i++)
            for (int j = 0; j < qEff; j++)
                hSubSym[i, j] = hSub[i, j] + hSub[j, i];
        // H_std = basis @ H_sub @ basisᵀ ; H = H_std / (sd_i sd_j)
        double[,] hStd = LinAlg.Dot(LinAlg.Dot(basis, hSubSym), Transpose(basis));
        var h = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                h[i, j] = hStd[i, j] / (sd[i] * sd[j]);
        var outv = PositiveHessianMetric(h, "pca_quadratic", fitR2);
        return outv is null ? null : new JanusMetricResult
        { JtJ = outv.JtJ, Mode = outv.Mode, Rank = outv.Rank, FitR2 = outv.FitR2, Q = qEff };
    }

    /// <summary>Combines diagonal quadratic features with a low-rank principal-component curvature model.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="sd">Per-coordinate scales used to standardize the numerical observations.</param>
    /// <param name="pcaK">Policy selecting the retained principal-component dimension.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>The fitted metric, or null when the local regression is not usable.</returns>
    private static JanusMetricResult? DiagPcaQuadratic(double[,] z, double[] y, double[] sd, string pcaK, double ridge)
    {
        var diag = FitDiagQuadratic(z, y, sd, ridge);
        var pca = FitPcaQuadratic(z, y, sd, pcaK, ridge);
        if (diag is null && pca is null) return LinearPseudo(z, y, sd, ridge);
        if (diag is null) return pca;
        if (pca is null || pca.Mode == "linear") return diag;
        int d = diag.JtJ.GetLength(0);
        var sum = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                sum[i, j] = (diag.JtJ[i, j] + pca.JtJ[i, j] + diag.JtJ[j, i] + pca.JtJ[j, i]) / 2.0;
        return new JanusMetricResult
        {
            JtJ = sum,
            Mode = "diag_pca_quadratic",
            Rank = diag.Rank + pca.Rank,
            FitR2 = Math.Max(diag.FitR2, pca.FitR2),
            Q = pca.Q,
        };
    }

    // ---- positive Hessian metric -----------------------------------------------------------------

    /// <summary>
    ///     Reference <c>_positive_hessian_metric</c>: symmetrize, eigendecompose, keep the positive
    ///     eigenpairs (relative threshold) and rebuild <c>V·diag(λ⁺)·Vᵀ</c>. Returns <c>null</c> when
    ///     no eigenvalue is positive (the numpy <c>None</c>).
    /// </summary>
    /// <param name="h">Fitted symmetric Hessian used to construct a nonnegative curvature metric.</param>
    /// <param name="mode">Feature-family name selecting the scalar local regression model.</param>
    /// <param name="fitR2">Coefficient of determination associated with the local fit.</param>
    /// <returns>A positive-semidefinite curvature metric from the fitted Hessian.</returns>
    private static JanusMetricResult? PositiveHessianMetric(double[,] h, string mode, double fitR2)
    {
        int d = h.GetLength(0);
        var hs = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                hs[i, j] = 0.5 * (h[i, j] + h[j, i]);
        if (!AllFinite(hs)) return null;
        double[] evals;
        double[,] evecs;
        try
        {
            (evals, evecs) = LinAlg.Eigh(hs);
        }
        catch (Exception)
        {
            return null;
        }
        double maxAbs = 0;
        foreach (double e in evals) maxAbs = Math.Max(maxAbs, Math.Abs(e));
        double thr = Math.Max(maxAbs, 1.0) * 1e-10;
        int rank = 0;
        for (int i = 0; i < d; i++) if (evals[i] > thr) rank++;
        if (rank == 0) return null;
        // JtJ = V⁺ diag(λ⁺) V⁺ᵀ  = Σ_pos λ_i v_i v_iᵀ
        var jtj = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            if (!(evals[i] > thr)) continue;
            double lam = evals[i];
            for (int a = 0; a < d; a++)
            {
                double va = evecs[a, i];
                if (va == 0) continue;
                for (int b = 0; b < d; b++)
                    jtj[a, b] += lam * va * evecs[b, i];
            }
        }
        var sym = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                sym[i, j] = 0.5 * (jtj[i, j] + jtj[j, i]);
        return new JanusMetricResult { JtJ = sym, Mode = mode, Rank = rank, FitR2 = fitR2 };
    }

    // ---- small numpy-shaped helpers --------------------------------------------------------------

    /// <summary>Solve <c>(MᵀM + ρI) w = Mᵀy</c>; <c>null</c> on failure / non-finite (numpy LinAlgError).</summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <param name="ridge">Nonnegative regularization added to stabilize the local linear solve.</param>
    /// <returns>Regularized regression coefficients, or null if the solve fails.</returns>
    private static double[]? RidgeSolve(double[,] m, double[] y, double ridge)
    {
        int cols = m.GetLength(1);
        double[,] gram = Gram(m);            // MᵀM  (cols×cols)
        for (int i = 0; i < cols; i++) gram[i, i] += ridge;
        double[] rhs = MatTVec(m, y);        // Mᵀy  (cols)
        try
        {
            double[] w = LinAlg.Solve(gram, rhs);
            return AllFinite(w) ? w : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>Measures explained variance of a fitted local regression model.</summary>
    /// <param name="phi">Regression design matrix with one observation per row.</param>
    /// <param name="coef">Fitted coefficients aligned with design-matrix columns.</param>
    /// <param name="y">Objective observations or second vector operand.</param>
    /// <returns>The coefficient of determination for the fitted observations.</returns>
    private static double FitR2(double[,] phi, double[] coef, double[] y)
    {
        int n = phi.GetLength(0);
        double[] yhat = LinAlg.Dot(phi, coef);
        double sst = 0, sse = 0;
        for (int i = 0; i < n; i++) { sst += y[i] * y[i]; double e = y[i] - yhat[i]; sse += e * e; }
        return sst <= Eps ? 0.0 : Math.Max(0.0, 1.0 - sse / sst);
    }

    /// <summary>Top-<paramref name="q" /> right-singular vectors of <c>Z</c> = top-q eigenvectors of ZᵀZ.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="q">Number of principal components to retain.</param>
    /// <returns>The leading right-singular-vector basis for the requested component count.</returns>
    private static double[,]? TopRightSingularVectors(double[,] z, int q)
    {
        int d = z.GetLength(1);
        double[,] gram = Gram(z);   // ZᵀZ (d×d)
        double[] evals;
        double[,] evecs;
        try
        {
            (evals, evecs) = LinAlg.Eigh(gram);   // ascending
        }
        catch (Exception)
        {
            return null;
        }
        int take = Math.Min(q, d);
        var basis = new double[d, take];
        // eigh returns ascending; take the largest `take` (columns d-1, d-2, …).
        for (int c = 0; c < take; c++)
        {
            int src = d - 1 - c;
            for (int r = 0; r < d; r++) basis[r, c] = evecs[r, src];
        }
        return basis;
    }

    /// <summary>Forms the column Gram matrix used in regularized least-squares fitting.</summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <returns>The symmetric column Gram matrix.</returns>
    private static double[,] Gram(double[,] m) => LinAlg.Dot(Transpose(m), m);

    /// <summary>Multiplies the transpose of a matrix by a vector.</summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <param name="v">Vector operand aligned with the matrix coordinates.</param>
    /// <returns>The product vector with one entry per matrix column.</returns>
    private static double[] MatTVec(double[,] m, double[] v)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var r = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            double s = 0;
            for (int i = 0; i < rows; i++) s += m[i, j] * v[i];
            r[j] = s;
        }
        return r;
    }

    /// <summary>Allocates a matrix with its row and column axes exchanged.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>A newly allocated transposed matrix.</returns>
    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j, i] = a[i, j];
        return t;
    }

    /// <summary>Allocates a diagonal matrix with the supplied vector on its diagonal.</summary>
    /// <param name="v">Vector operand aligned with the matrix coordinates.</param>
    /// <returns>A square matrix whose off-diagonal entries are zero.</returns>
    private static double[,] Diag(double[] v)
    {
        int n = v.Length;
        var m = new double[n, n];
        for (int i = 0; i < n; i++) m[i, i] = v[i];
        return m;
    }

    /// <summary>Lists upper-triangular coordinate pairs for quadratic interaction features.</summary>
    /// <param name="d">Decision-space dimension.</param>
    /// <returns>Row and column index arrays for the upper triangle.</returns>
    private static (int[] Row, int[] Col) TriuIndices(int d)
    {
        int pairs = d * (d + 1) / 2;
        var row = new int[pairs];
        var col = new int[pairs];
        int k = 0;
        for (int i = 0; i < d; i++)
            for (int j = i; j < d; j++) { row[k] = i; col[k] = j; k++; }
        return (row, col);
    }

    /// <summary>Produces indices that order values by ascending magnitude of their stored numeric value.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>Indices ordered by ascending value.</returns>
    private static int[] ArgSort(double[] a)
    {
        var idx = new int[a.Length];
        for (int i = 0; i < a.Length; i++) idx[i] = i;
        Array.Sort(idx, (i, j) => a[i].CompareTo(a[j]));   // stable enough for tie order here
        return idx;
    }

    /// <summary>Subtracts the vector mean from every entry.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>A centered copy of the vector.</returns>
    private static double[] Center(double[] a)
    {
        double mean = 0;
        foreach (double v in a) mean += v;
        mean /= a.Length;
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] - mean;
        return r;
    }

    /// <summary>Finds the smallest vector entry using the reference reduction semantics.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The smallest vector entry.</returns>
    private static double Min(double[] a) { double m = a[0]; foreach (double v in a) if (v < m) m = v; return m; }
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
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (!IsFinite(a[i, j])) return false;
        return true;
    }
}

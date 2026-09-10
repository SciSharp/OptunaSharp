// JANUS infill driver — the host-agnostic core of the reference `_JanusOptimizer`
// (`janus_optuna/janus.py`): the window harvest, the Jacobian assembly (GN elite + JᵀJ), the CMIX
// exploration metric, the bounded injection, and the ask/tell + coherent-generation batching loop —
// all expressed against an `IPopulationHost` rather than a concrete CMA-ES. The CMA-ES specifics the
// Phase-1 optimizer entangled here (host construction, native ask/tell, and the mean/σ/covariance
// proposal statistics) now live behind `CmaPopulationHost`; DE and MOO hosts (Plan 05 Phases 2/3)
// plug into the same seam (Plan 03). This extraction is behavior-preserving: the Phase-1 JANUS tests
// are the contract, and the CMA-ES-hosted candidate/elite stream is byte-for-byte unchanged.
//
// Reference: JANUS — "Online Jacobian-Aligned Infill for Black-Box Optimization", Hongyuan Yu et al.,
// arXiv:2608.22862 (2026); https://github.com/hongyuanyu/JANUS (Apache-2.0). JANUS is NOT part of
// Optuna → this is a native paper-faithful port validated by convergence (same footing as `LmCma`),
// no upstream trajectory to bit-match.
//
// Documented divergences from the reference (all principled, none change the algorithm):
//   • Host = an `IPopulationHost` (default `CmaPopulationHost`, MT19937 RNG), not evotorch/torch — so
//     the whole stack obeys the project's "never System.Random, always MT19937" RNG discipline. CMIX
//     noise uses a second MT19937 `NumPyRandom`, mirroring the reference's separate numpy stream.
//   • Coordinate transform = OptunaSharp `SearchSpaceTransform(transform01)` (the box the host
//     expects), not the reference's `dynamic_normalize_params` — pure host plumbing, not JANUS math.
//   • Bounded injection uses the square-root-invariant Mahalanobis clip (‖z‖ = √(dᵀC⁻¹d)/σ via
//     `LinAlg.Solve(C,d)`); when clipped the step is a radial scaling of (x_e − mean), identical for
//     any covariance square root (the reference's Cholesky-whitened clip reduces to the same thing).
//   • The reference gates its window on trials carrying a `component_losses` residual vector; we ALSO
//     harvest scalar-only trials and drive JANUS from the scalar `pseudo_metric` (the paper's BBOB
//     "scalar linear/quadratic residual coordinate" mode — machinery already present in the reference).
//   • cmix_mode "reshape" (in-place host-covariance mixing, `CmixShapeC`) and cmix_candidate_center
//     "archive" (`ArchiveCenters`) are both ported; a covariance-free host (DE) makes reshape a no-op.
//     The reference DEFAULT stays "portfolio" (CMIX candidates in tail slots) + center "mean".

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Janus;

/// <summary>
///     One in-memory JANUS infill driver: an <see cref="IPopulationHost" /> plus the host-agnostic
///     Jacobian-infill logic (window harvest, Gauss–Newton elite, CMIX exploration, bounded injection,
///     and coherent-generation batching). Held live by <see cref="JanusSampler" /> for the study's
///     duration (like the reference's <c>optimizer_dict</c>); it re-reads the completed trials each
///     call rather than serializing state. The CMA-ES host specifics live behind
///     <see cref="CmaPopulationHost" />; DE and MOO hosts (Plan 05) plug into the same seam.
/// </summary>
internal sealed class JanusOptimizer
{
    private readonly SearchSpaceTransform _trans;
    private readonly StudyDirection _direction;
    private readonly int _d;
    private readonly int _popsize;
    private readonly double _sqrtD;
    private readonly double[] _lo;
    private readonly double[] _hi;
    private readonly IPopulationHost _host;
    private readonly NumPyRandom _rng;

    // hyperparameters (null string/int ⇒ "auto")
    private readonly int _fuseEvery;
    private readonly int? _fuseStart;
    private readonly double _kappa;
    private readonly int? _gnWindow;
    private readonly double _gnLambda;
    private readonly double _gnRidge;
    private readonly double? _cmixW;
    private readonly double _cmixLambda;
    private readonly double _cmixWBase;
    private readonly double _cmixWMax;
    private readonly string _gnPseudoMode;
    private readonly string _cmixPseudoMode;
    private readonly string _pseudoTarget;
    private readonly string _pcaK;
    private readonly string _cmixMode;
    private readonly double _cmixCandidateFrac;
    private readonly string _cmixCandidateCenter;

    // evolving state
    private readonly List<double[]> _winX = new();
    private readonly List<double[]?> _winR = new();
    private readonly List<double> _winF = new();
    private readonly HashSet<int> _seen = new();
    private readonly List<int> _trialsUsed = new();
    // Trials JANUS actually generated a candidate for, in ask order. The host is told only these,
    // grouped into coherent generations of `popsize` — startup trials feed the window (Harvest) but
    // are NOT told as host generations. This avoids the reference's startup-remainder batch
    // misalignment (its sequential batching of ALL completed trials mixes samples from different
    // distributions into one CMA update whenever nStartupTrials is not a multiple of popsize).
    private readonly List<int> _askedOrder = new();
    private readonly HashSet<int> _asked = new();
    private int _generation;
    private double[][]? _pending;
    private double[,]? _lastJtJ;
    private double? _cmixWEff;
    private int _archiveCursor;   // round-robins the archive centers across generations (cmix_candidate_center="archive")

    private List<FrozenTrial> _completedTrials;

    internal JanusOptimizer(
        SearchSpaceTransform trans, IPopulationHost host, List<FrozenTrial> completedTrials, StudyDirection direction,
        int cmixSeed,
        int fuseEvery, int? fuseStart, double kappa,
        int? gnWindow, double gnLambda, double gnRidge,
        double? cmixW, double cmixLambda, double cmixWBase, double cmixWMax,
        string gnPseudoMode, string cmixPseudoMode, string pseudoTarget,
        string pcaK, string cmixMode, double cmixCandidateFrac, string cmixCandidateCenter)
    {
        _trans = trans;
        _host = host;
        _completedTrials = completedTrials;
        _direction = direction;
        _fuseEvery = Math.Max(1, fuseEvery);
        _fuseStart = fuseStart;
        _kappa = kappa;
        _gnWindow = gnWindow;
        _gnLambda = gnLambda;
        _gnRidge = gnRidge;
        _cmixW = cmixW;
        _cmixLambda = cmixLambda;
        _cmixWBase = cmixWBase;
        _cmixWMax = cmixWMax;
        _gnPseudoMode = gnPseudoMode;
        _cmixPseudoMode = cmixPseudoMode;
        _pseudoTarget = pseudoTarget;
        _pcaK = pcaK;
        _cmixMode = (cmixMode ?? "portfolio").ToLowerInvariant();
        _cmixCandidateFrac = cmixCandidateFrac;
        _cmixCandidateCenter = (cmixCandidateCenter ?? "mean").ToLowerInvariant();

        // The driver derives only the box bounds it needs for candidate clipping and κ√d from the
        // transform; the host owns the proposal (mean/step/covariance) and the population size λ.
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        _d = boundsFlat.Length / 2;
        _lo = new double[_d];
        _hi = new double[_d];
        for (int i = 0; i < _d; i++)
        {
            _lo[i] = boundsFlat[i * 2];
            _hi[i] = boundsFlat[i * 2 + 1];
        }
        _sqrtD = Math.Sqrt(_d);
        _popsize = host.PopulationSize;

        _rng = np.random.RandomState();
        _rng.seed((uint)(cmixSeed ^ 0x5A17));   // second MT19937 stream for CMIX noise (≠ the host stream)
    }

    internal int PopulationSize => _popsize;
    internal int Dim => _d;

    internal void SetCompletedTrials(List<FrozenTrial> completed) => _completedTrials = completed;

    // ---- window harvesting -----------------------------------------------------------------------

    private void Harvest()
    {
        foreach (var t in _completedTrials)
        {
            if (t.Number < 0 || _seen.Contains(t.Number)) continue;
            // The observed-components channel (component_losses / rae user-attr) — read through the
            // single shared reader so this window harvest and the MOO metric never drift apart.
            double[]? rae = t.TryGetComponentLosses(out var comp) ? comp : null;
            double[] x;
            try
            {
                x = _trans.Transform(t.ParamsDict).ToArray<double>();
            }
            catch (Exception)
            {
                continue;
            }
            double fval;
            if (t.Value is not null)
                fval = _direction == StudyDirection.Maximize ? -t.Value.Value : t.Value.Value;
            else if (rae is not null)
                fval = rae.Average();
            else
                continue;
            _winX.Add(x);
            _winR.Add(rae);
            _winF.Add(fval);
            _seen.Add(t.Number);
        }
        int? m = LastComponentWidth();
        int window = ResolveGnWindow(m);
        TrimWindow(window);
    }

    private void TrimWindow(int window)
    {
        int extra = _winX.Count - window;
        if (extra <= 0) return;
        _winX.RemoveRange(0, extra);
        _winR.RemoveRange(0, extra);
        _winF.RemoveRange(0, extra);
    }

    private int? LastComponentWidth()
    {
        for (int i = _winR.Count - 1; i >= 0; i--)
            if (_winR[i] is { } r) return r.Length;
        return null;
    }

    // ---- resolve auto hyperparameters ------------------------------------------------------------

    private int ResolveGnWindow(int? m)
    {
        if (_gnWindow is int set) return Math.Max(_popsize, set);
        int minWindow = Math.Max(4 * _popsize, 80);
        int target = m is null ? Math.Max(minWindow, 300) : Math.Max(minWindow, 3 * m.Value);
        return Math.Min(Math.Max(target, minWindow), 1200);
    }

    private int ResolveFuseStart(int window)
    {
        if (_fuseStart is int set) return Math.Max(_popsize, Math.Min(set, window));
        return Math.Max(2 * _popsize, Math.Min(200, Math.Max(_popsize, window / 2)));
    }

    private double ResolveCmixW(int m)
    {
        if (_cmixWEff is double eff) return eff;
        double w;
        if (_cmixW is null)   // "auto"
        {
            double r = Math.Min(m, ResolveGnWindow(m));
            w = _cmixWBase * r / _d;
            double autoMin = _cmixWBase / _d;
            w = Math.Min(Math.Max(w, autoMin), _cmixWMax);
        }
        else
        {
            w = _cmixW.Value;
        }
        _cmixWEff = w;
        return w;
    }

    // ---- GN/LM elite + cache JᵀJ for CMIX --------------------------------------------------------

    /// <summary>
    ///     Component-based residual Jacobian fit — a verbatim port of the reference <c>_assemble_gn</c>
    ///     lines (janus.py): <c>S = √max(R,0)</c>; <c>Sc = S − S.mean(0)</c>;
    ///     <c>AinvZt = solve(ZᵀZ + ρI, Zᵀ)</c>; <c>W = (AinvZt·Sc)ᵀ</c>; <c>J = W / sd</c>;
    ///     <c>G = JᵀJ</c>. Returns <c>(J, G)</c>, or <c>null</c> if the ridge solve fails. Extracted so
    ///     it is a single source of truth shared by <see cref="AssembleGn" /> and the byte-parity
    ///     interop test (the live <c>_assemble_gn</c> is unreachable without torch). <c>A⁻¹Zᵀ·Sc</c> is
    ///     computed as <c>solve(A, Zᵀ·Sc)</c> (algebraically identical, fewer RHS columns).
    /// </summary>
    internal static (double[,] J, double[,] GnJtJ)? FitComponentJacobian(
        double[,] x, double[,] rComponents, double gnRidge)
    {
        int n = x.GetLength(0), d = x.GetLength(1), m = rComponents.GetLength(1);
        var (z, sd) = Standardize(x);
        // S = √max(R,0); center each column (S − S.mean(0)).
        var s = new double[n, m];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < m; c++) s[i, c] = Math.Sqrt(Math.Max(rComponents[i, c], 0.0));
        for (int c = 0; c < m; c++)
        {
            double t = 0;
            for (int i = 0; i < n; i++) t += s[i, c];
            double mean = t / n;
            for (int i = 0; i < n; i++) s[i, c] -= mean;
        }
        double[,] gram = LinAlg.Dot(Transpose(z), z);   // ZᵀZ + ρI
        for (int i = 0; i < d; i++) gram[i, i] += gnRidge;
        double[,] ztsc = LinAlg.Dot(Transpose(z), s);   // Zᵀ Sc  (d×m)
        double[,]? ainvZtSc = SolveMatrix(gram, ztsc);  // A⁻¹ Zᵀ Sc  (d×m)
        if (ainvZtSc is null) return null;
        var jac = new double[m, d];                     // J = (A⁻¹ Zᵀ Sc)ᵀ / sd
        for (int c = 0; c < m; c++)
            for (int col = 0; col < d; col++)
                jac[c, col] = ainvZtSc[col, c] / sd[col];
        double[,] gnJtJ = LinAlg.Dot(Transpose(jac), jac);   // JᵀJ  (d×d)
        return (jac, gnJtJ);
    }

    private double[]? AssembleGn()
    {
        _lastJtJ = null;
        int? mLast = LastComponentWidth();
        int window = ResolveGnWindow(mLast);
        int fuseStart = ResolveFuseStart(window);
        if (_winX.Count < Math.Max(fuseStart, _popsize)) return null;

        int n = _winX.Count;
        var x = new double[n, _d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < _d; j++) x[i, j] = _winX[i][j];
        var f = _winF.ToArray();

        bool hasComponents = mLast is int mw && mw > 0 && _winR.All(r => r is { } rr && rr.Length == mw);
        int m = mLast ?? 0;

        // Standardize (population std + 1e-9), matching JanusMetric / the reference _standardize.
        var (z, sd) = Standardize(x);
        // Gram A = ZᵀZ + gn_ridge·I  (reused by the component fit and the pseudo scalar gradient).
        double[,] gram = LinAlg.Dot(Transpose(z), z);
        for (int i = 0; i < _d; i++) gram[i, i] += _gnRidge;

        double[,]? jac = null;      // component Jacobian (m×d)
        double[,]? gnJtJ = null;
        if (hasComponents)
        {
            var rMat = new double[n, m];
            for (int i = 0; i < n; i++)
                for (int c = 0; c < m; c++) rMat[i, c] = _winR[i]![c];
            var fit = FitComponentJacobian(x, rMat, _gnRidge);
            if (fit is null) return null;
            (jac, gnJtJ) = fit.Value;
            ResolveCmixW(m);
        }

        var pseudo = JanusMetric.Compute(x, f, _gnPseudoMode, _gnRidge, _pcaK, _pseudoTarget);
        // Reference: use_pseudo = pseudo != None and mode != "linear". Scalar extension: with no
        // components there is no component J, so a linear pseudo metric is accepted too.
        bool usePseudo = pseudo is not null && (pseudo.Mode != "linear" || !hasComponents);

        double[,] lastJtJ;
        if (usePseudo)
            lastJtJ = pseudo!.JtJ;
        else if (hasComponents)
            lastJtJ = gnJtJ!;
        else
            return null;
        _lastJtJ = lastJtJ;

        // best point in the window
        int ib;
        double[]? sB = null;
        if (hasComponents)
        {
            ib = 0;
            double best = double.PositiveInfinity;
            for (int i = 0; i < n; i++)
            {
                double mean = 0;
                for (int c = 0; c < m; c++) mean += _winR[i]![c];
                mean /= m;
                if (mean < best) { best = mean; ib = i; }
            }
            // s_b is the UNCENTERED √max(R_ib,0) (the reference centers S only for the J regression;
            // s_b = S[ib] uses the raw column).
            sB = new double[m];
            for (int c = 0; c < m; c++) sB[c] = Math.Sqrt(Math.Max(_winR[ib]![c], 0.0));
        }
        else
        {
            ib = ArgMin(f);
        }

        double[,] h = (double[,])lastJtJ.Clone();
        for (int i = 0; i < _d; i++) h[i, i] += _gnLambda;

        double[] delta;
        double[]? g = null;
        try
        {
            if (usePseudo)
            {
                double[] y = JanusMetric.PseudoTarget(f, _pseudoTarget);
                double[] zty = MatTVec(z, y);
                double[] w = LinAlg.Solve(gram, zty);
                g = new double[_d];
                for (int i = 0; i < _d; i++) g[i] = w[i] / sd[i];
                double[] gd = LinAlg.Solve(h, g);
                delta = Negate(gd);
            }
            else
            {
                double[] jts = MatTVec(jac!, sB!);   // Jᵀ s_b (d)
                double[] gd = LinAlg.Solve(h, jts);
                delta = Negate(gd);
            }
        }
        catch (Exception)
        {
            return null;
        }
        if (!AllFinite(delta)) return null;

        // line search over the three fixed steps, pick the min predicted decrease.
        double bestT = 1.0, bestPred = double.PositiveInfinity;
        foreach (double step in new[] { 0.25, 0.5, 1.0 })
        {
            double pred;
            if (usePseudo)
            {
                double gd = Dot(g!, delta);
                double dHd = Dot(delta, MatVec(lastJtJ, delta));
                pred = _winF[ib] + step * gd + 0.5 * step * step * dHd;
            }
            else
            {
                double[] jd = MatVec(jac!, delta);   // J·delta (m)
                double sum = 0;
                for (int c = 0; c < m; c++) { double r = sB![c] + step * jd[c]; sum += r * r; }
                pred = sum;
            }
            if (pred < bestPred) { bestPred = pred; bestT = step; }
        }

        var elite = new double[_d];
        for (int i = 0; i < _d; i++) elite[i] = Clip(_winX[ib][i] + bestT * delta[i], _lo[i], _hi[i]);
        return elite;
    }

    // ---- CMIX candidate metric + samples ---------------------------------------------------------

    private (double[,] Eigvecs, double[] Eigvals)? CandidateMetric()
    {
        if (_lastJtJ is null) return null;
        double[,] jtj = _lastJtJ;
        if (_cmixPseudoMode != _gnPseudoMode && _winX.Count > 0)
        {
            int n = _winX.Count;
            var x = new double[n, _d];
            for (int i = 0; i < n; i++) for (int c = 0; c < _d; c++) x[i, c] = _winX[i][c];
            var cmix = JanusMetric.Compute(x, _winF.ToArray(), _cmixPseudoMode, _gnRidge, _pcaK, _pseudoTarget);
            if (cmix is not null && cmix.Mode != "linear")
            {
                jtj = cmix.JtJ;
                _lastJtJ = jtj;
            }
        }
        // metric = inv(JᵀJ + λ_mix I). Since it is symmetric PD, eigh(JᵀJ+λI)=(μ_i,v_i) and the
        // metric's eigenpairs are (1/μ_i, v_i) — no explicit inverse needed.
        var hs = (double[,])jtj.Clone();
        for (int i = 0; i < _d; i++) hs[i, i] += _cmixLambda;
        for (int i = 0; i < _d; i++) for (int k = i + 1; k < _d; k++) { double a = 0.5 * (hs[i, k] + hs[k, i]); hs[i, k] = a; hs[k, i] = a; }
        double[] mu;
        double[,] eigvecs;
        try
        {
            (mu, eigvecs) = LinAlg.Eigh(hs);
        }
        catch (Exception)
        {
            return null;
        }
        var eigvals = new double[_d];
        for (int i = 0; i < _d; i++)
        {
            double val = mu[i] > 0 ? 1.0 / mu[i] : double.PositiveInfinity;
            eigvals[i] = Math.Max(val, 1e-12);
            if (!IsFinite(eigvals[i])) return null;
        }
        double trC = _host.ProposalCovarianceTrace;
        double trM = 0; foreach (double e in eigvals) trM += e;
        if (trC <= 0.0 || trM <= 0.0 || !IsFinite(trC)) return null;
        double scale = trC / trM;
        for (int i = 0; i < _d; i++) eigvals[i] *= scale;
        return (eigvecs, eigvals);
    }

    // ---- cmix_mode "reshape": in-place host-covariance mixing ------------------------------------

    /// <summary>
    ///     Port of the reference <c>_cmix_shape_C</c>: mix a trace-matched inverse-curvature term into the
    ///     host's proposal covariance, <c>C ← (1−w)·C + w·(tr(C)/tr(T))·T</c> with <c>T = (JᵀJ + λ_mix I)⁻¹</c>.
    ///     No-op unless the effective CMIX weight <c>w</c> is set (component mode) and the host exposes a
    ///     covariance (a covariance-free DE host makes this inert). The host re-derives its decomposition.
    /// </summary>
    private void CmixShapeC()
    {
        double? w = _cmixWEff;
        if (_lastJtJ is null || w is null || w.Value <= 0.0) return;
        if (!_host.HasProposalCovariance) return;

        int d = _d;
        var a = (double[,])_lastJtJ.Clone();     // JᵀJ + λ_mix I
        for (int i = 0; i < d; i++) a[i, i] += _cmixLambda;
        double[,]? t = InvertMatrix(a);          // T = inv(JᵀJ + λ_mix I)
        if (t is null || !AllFinite(t)) return;
        for (int i = 0; i < d; i++)               // symmetrize T
            for (int k = i + 1; k < d; k++) { double v = 0.5 * (t[i, k] + t[k, i]); t[i, k] = v; t[k, i] = v; }

        double[,]? c = _host.GetProposalCovariance();
        if (c is null) return;
        double trC = 0, trT = 0;
        for (int i = 0; i < d; i++) { trC += c[i, i]; trT += t[i, i]; }
        if (trT <= 1e-30 || !IsFinite(trC)) return;

        double scale = trC / trT;
        double wv = w.Value;
        var cNew = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int k = 0; k < d; k++)
                cNew[i, k] = (1.0 - wv) * c[i, k] + wv * (t[i, k] * scale);
        var sym = new double[d, d];               // symmetrize C_new
        for (int i = 0; i < d; i++)
            for (int k = 0; k < d; k++)
                sym[i, k] = 0.5 * (cNew[i, k] + cNew[k, i]);
        _host.TrySetProposalCovariance(sym);
    }

    private static double[,]? InvertMatrix(double[,] a)
    {
        int n = a.GetLength(0);
        var id = new double[n, n];
        for (int i = 0; i < n; i++) id[i, i] = 1.0;
        return SolveMatrix(a, id);
    }

    // ---- CMIX candidate metric + samples ---------------------------------------------------------

    private double[][]? CmixCandidateSamples(int count, double[]? elite)
    {
        // A host without a proposal covariance offers no CMIX candidates (the paper's GN-only infill).
        if (count <= 0 || _lastJtJ is null || !_host.HasProposalCovariance) return null;
        var metric = CandidateMetric();
        if (metric is null) return null;
        var (eigvecs, eigvals) = metric.Value;

        var sqrtEig = new double[_d];
        for (int i = 0; i < _d; i++) sqrtEig[i] = Math.Sqrt(eigvals[i]);

        // cmix_candidate_center="archive": one sample per Pareto-archive center (round-robined via
        // _archiveCursor), rather than `count` samples around one center — the reference
        // `_cmix_candidate_samples` archive branch. Empty archive ⇒ fall through to the single-center path.
        if (_cmixCandidateCenter == "archive")
        {
            var centers = ArchiveCenters(count);
            if (centers.Count > 0)
            {
                double sig = Math.Max(_host.Sigma, 1e-10);
                var archived = new double[centers.Count][];
                for (int k = 0; k < centers.Count; k++)
                {
                    double[] zk = _rng.randn(_d).ToArray<double>();
                    var nk = new double[_d];
                    for (int i = 0; i < _d; i++) nk[i] = zk[i] * sqrtEig[i] * sig;   // normal · √eig · σ
                    double[] y = MatVec(eigvecs, nk);                                // eigvecs @ noise
                    var s = new double[_d];
                    for (int i = 0; i < _d; i++) s[i] = Clip(centers[k][i] + y[i], _lo[i], _hi[i]);
                    archived[k] = s;
                }
                return archived;
            }
        }

        double[] center;
        if (_cmixCandidateCenter == "best" && _winX.Count > 0)
            center = (double[])_winX[ArgMin(_winF.ToArray())].Clone();
        else if (_cmixCandidateCenter == "elite" && elite is not null)
            center = (double[])elite.Clone();
        else
            center = (double[])_host.Mean.Clone();

        double sigScale = _host.Sigma;

        var samples = new double[count][];
        for (int k = 0; k < count; k++)
        {
            double[] zk = _rng.randn(_d).ToArray<double>();
            var nk = new double[_d];
            for (int i = 0; i < _d; i++) nk[i] = zk[i] * sqrtEig[i];
            double[] y = MatVec(eigvecs, nk);                 // eigvecs @ (√eigvals ⊙ z)
            var s = new double[_d];
            for (int i = 0; i < _d; i++) s[i] = Clip(center[i] + sigScale * y[i], _lo[i], _hi[i]);
            samples[k] = s;
        }
        return samples;
    }

    /// <summary>
    ///     Port of the reference <c>_archive_centers</c>: from the residual-component window take the
    ///     best-by-mean-residual slice (truncated to <c>max(200, 8·count)</c>), keep its dominance-filtered
    ///     set, farthest-point-sample a diverse subset in normalized-residual space, and round-robin
    ///     <paramref name="count" /> centers through <see cref="_archiveCursor" />. Requires every window row
    ///     to carry components (else an empty list). NOTE: as with <c>_nondominated_indices</c> in the MOO
    ///     file, the dominance filter keeps the DOMINATED points — ported verbatim.
    /// </summary>
    private List<double[]> ArchiveCenters(int count)
    {
        var empty = new List<double[]>();
        if (count <= 0 || _winX.Count == 0) return empty;

        int m = -1;
        foreach (var r in _winR)
        {
            if (r is null) return empty;
            if (m < 0) m = r.Length;
            else if (r.Length != m) return empty;
        }
        if (m <= 0) return empty;

        var finite = new List<int>();
        for (int i = 0; i < _winX.Count; i++)
        {
            bool ok = true;
            for (int j = 0; ok && j < _d; j++) ok = IsFinite(_winX[i][j]);
            for (int c = 0; ok && c < m; c++) ok = IsFinite(_winR[i]![c]);
            if (ok) finite.Add(i);
        }
        if (finite.Count == 0) return empty;

        double MeanR(int i) { double s = 0; for (int c = 0; c < m; c++) s += _winR[i]![c]; return s / m; }
        var order = finite.OrderBy(MeanR).ToList();   // LINQ OrderBy is stable ⇒ mergesort-equivalent
        int trunc = Math.Min(order.Count, Math.Max(200, 8 * count));
        order = order.Take(trunc).ToList();

        int nt = order.Count;
        var rMat = new double[nt][];
        var xMat = new double[nt][];
        for (int k = 0; k < nt; k++) { rMat[k] = _winR[order[k]]!; xMat[k] = _winX[order[k]]; }

        // reference-quirk "non-dominated" filter (keeps the DOMINATED points; see _nondominated_indices)
        var keep = new bool[nt];
        for (int k = 0; k < nt; k++) keep[k] = true;
        for (int i = 0; i < nt; i++)
        {
            if (!keep[i]) continue;
            for (int j = 0; j < nt; j++)
            {
                bool allLe = true, anyLt = false;
                for (int c = 0; c < m; c++)
                {
                    if (rMat[j][c] > rMat[i][c]) { allLe = false; break; }
                    if (rMat[j][c] < rMat[i][c]) anyLt = true;
                }
                if (allLe && anyLt && j != i) keep[j] = false;
            }
        }
        var archiveX = new List<double[]>();
        var archiveR = new List<double[]>();
        for (int k = 0; k < nt; k++) if (keep[k]) { archiveX.Add(xMat[k]); archiveR.Add(rMat[k]); }
        if (archiveX.Count == 0) return empty;

        if (archiveX.Count > 1)
        {
            int na = archiveX.Count;
            var lo = new double[m];
            var hi = new double[m];
            for (int c = 0; c < m; c++) { lo[c] = double.PositiveInfinity; hi[c] = double.NegativeInfinity; }
            foreach (var r in archiveR)
                for (int c = 0; c < m; c++) { if (r[c] < lo[c]) lo[c] = r[c]; if (r[c] > hi[c]) hi[c] = r[c]; }
            var normR = new double[na][];
            for (int k = 0; k < na; k++)
            {
                normR[k] = new double[m];
                for (int c = 0; c < m; c++) normR[k][c] = (archiveR[k][c] - lo[c]) / Math.Max(hi[c] - lo[c], 1e-12);
            }
            // farthest-point sampling seeded by the min-mean-normalized point
            int first = 0; double best = double.PositiveInfinity;
            for (int k = 0; k < na; k++)
            {
                double s = 0; for (int c = 0; c < m; c++) s += normR[k][c]; s /= m;
                if (s < best) { best = s; first = k; }
            }
            var selected = new List<int> { first };
            var isSel = new bool[na]; isSel[first] = true;
            int need = Math.Min(count, na);
            while (selected.Count < need)
            {
                int bestK = -1; double bestDist = double.NegativeInfinity;
                for (int k = 0; k < na; k++)
                {
                    if (isSel[k]) continue;
                    double dmin = double.PositiveInfinity;
                    foreach (int sIdx in selected)
                    {
                        double dd = 0;
                        for (int c = 0; c < m; c++) { double diff = normR[k][c] - normR[sIdx][c]; dd += diff * diff; }
                        dd = Math.Sqrt(dd);
                        if (dd < dmin) dmin = dd;
                    }
                    if (dmin > bestDist) { bestDist = dmin; bestK = k; }
                }
                if (bestK < 0) break;
                selected.Add(bestK); isSel[bestK] = true;
            }
            var sel = new List<double[]>();
            foreach (int s in selected) sel.Add(archiveX[s]);
            archiveX = sel;
        }

        int len = archiveX.Count;
        var centers = new List<double[]>();
        for (int i = 0; i < count; i++) centers.Add((double[])archiveX[(_archiveCursor + i) % len].Clone());
        _archiveCursor = (_archiveCursor + count) % Math.Max(1, len);
        return centers;
    }

    // ---- bounded injection (square-root-invariant Mahalanobis clip) ------------------------------

    private double[] BoundedElite(double[] fused)
    {
        double[] mean = _host.Mean;
        var diff = new double[_d];
        for (int i = 0; i < _d; i++) diff[i] = fused[i] - mean[i];
        // ‖z‖ of the step in the host's proposal metric — CMA-ES: √(diffᵀC⁻¹diff)/σ. A value ≤ 0 means
        // "no whitening available" (a covariance-free host, or a singular solve), and the candidate is
        // injected unclipped, then box-clamped — the reference's Cholesky-whitened clip reduces to this.
        double znorm = _host.WhitenedStepNorm(diff);
        double cap = _kappa * _sqrtD;
        var x = new double[_d];
        if (znorm > cap && znorm > 1e-12)
        {
            double scale = cap / znorm;
            for (int i = 0; i < _d; i++) x[i] = mean[i] + scale * diff[i];
        }
        else
        {
            for (int i = 0; i < _d; i++) x[i] = fused[i];
        }
        for (int i = 0; i < _d; i++) x[i] = Clip(x[i], _lo[i], _hi[i]);
        return x;
    }

    // ---- generation + ask/tell -------------------------------------------------------------------

    private void Generate()
    {
        double[]? elite = null;
        if (_generation % _fuseEvery == 0)
        {
            elite = AssembleGn();
            // cmix_mode "reshape": mix the inverse-curvature term into the host covariance in place
            // (the reference's `_cmix_shape_C`, called right after `_assemble_gn`). A covariance-free host
            // (DE) makes this a no-op.
            if (_cmixMode == "reshape") CmixShapeC();
        }

        // Native candidates use the host's own feasibility handling (Cma.Ask = resample-then-repair),
        // exactly as CmaEsSampler draws from the same Cma — so with infill off JANUS reduces to plain
        // CMA-ES. (The reference hard-clamps its evotorch samples; hard-clamping piles mass on the box
        // boundary and biases the covariance update, so we use the host's resampling instead — only the
        // deliberately-constructed injected candidates below are box-clipped.)
        var xs = new double[_popsize][];
        for (int k = 0; k < _popsize; k++)
            xs[k] = _host.Ask();

        if (_cmixMode is "candidates" or "candidate" or "portfolio" or "hybrid")
        {
            // Python round() is round-half-to-even; C#'s default Math.Round matches it (do NOT use
            // AwayFromZero — it diverges at exact halves, e.g. popsize·frac = 0.5 → Python 0, not 1).
            int nCand = (int)Math.Round(_popsize * _cmixCandidateFrac, MidpointRounding.ToEven);
            nCand = Math.Max(0, Math.Min(_popsize - 1, nCand));
            double[][]? cand = CmixCandidateSamples(nCand, elite);
            if (cand is { Length: > 0 })
            {
                int start = _popsize - cand.Length;   // tail slots
                for (int k = 0; k < cand.Length; k++) xs[start + k] = cand[k];
            }
        }

        if (elite is not null)
            xs[0] = BoundedElite(elite);

        _pending = xs;
    }

    internal OrderedDict<string, object?> Ask(int trialNumber)
    {
        if (_pending is null) Generate();
        // Record this trial as a JANUS-generated one so UpdateEs can tell it as part of a coherent
        // generation. `idx = trialNumber % popsize` picks the slot (GN elite = slot 0), matching the
        // reference; popsize consecutive asked trials consume exactly one pending generation.
        if (_asked.Add(trialNumber)) _askedOrder.Add(trialNumber);
        double[] x = _pending![trialNumber % _popsize];
        var safe = new double[_d];
        for (int i = 0; i < _d; i++)
        {
            double v = x[i];
            // Non-finite candidate (a failed fit slipped through) → fall back to a uniform box draw,
            // mirroring the reference's `x = lo + rng.random(d)*(hi-lo)` guard.
            if (!IsFinite(v)) v = _lo[i] + (_hi[i] - _lo[i]) * _rng.uniform(0.0, 1.0).ToArray<double>()[0];
            safe[i] = Clip(v, _lo[i], _hi[i]);
        }
        return _trans.Untransform(np.array(safe));
    }

    internal void UpdateEs()
    {
        if (_completedTrials.Count == 0) return;
        Harvest();

        // Tell the host only coherent generations: the JANUS-asked trials that have completed, in ask
        // order, grouped by popsize. Each such group is exactly one pending generation (popsize
        // consecutive asked trials consume one `_generate()` batch), so every CMA update sees samples
        // from a single distribution — regardless of nStartupTrials.
        var byNumber = new Dictionary<int, FrozenTrial>(_completedTrials.Count);
        foreach (var t in _completedTrials) byNumber[t.Number] = t;
        var used = new HashSet<int>(_trialsUsed);
        var ready = new List<FrozenTrial>();
        foreach (int number in _askedOrder)
        {
            if (used.Contains(number)) continue;
            if (byNumber.TryGetValue(number, out var t) && t.Value is not null) ready.Add(t);
            else break;   // preserve ask order; wait for an earlier trial to complete before batching later ones
        }

        int genBefore = _generation;
        int batches = ready.Count / _popsize;
        for (int b = 0; b < batches; b++)
        {
            var solutions = new List<(double[] X, double Value)>(_popsize);
            for (int k = 0; k < _popsize; k++)
            {
                var t = ready[b * _popsize + k];
                double[] x = _trans.Transform(t.ParamsDict).ToArray<double>();
                double y = _direction == StudyDirection.Maximize ? -t.Value!.Value : t.Value!.Value;
                solutions.Add((x, y));
                _trialsUsed.Add(t.Number);
            }
            _host.Tell(solutions);
            _generation += 1;
        }
        if (_generation != genBefore) _pending = null;
    }

    // ---- small numpy-shaped helpers --------------------------------------------------------------

    private static (double[,] Z, double[] Sd) Standardize(double[,] x)
    {
        int n = x.GetLength(0), d = x.GetLength(1);
        var mu = new double[d];
        var sd = new double[d];
        for (int j = 0; j < d; j++) { double s = 0; for (int i = 0; i < n; i++) s += x[i, j]; mu[j] = s / n; }
        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) { double diff = x[i, j] - mu[j]; s += diff * diff; }
            sd[j] = Math.Sqrt(s / n) + 1e-9;
        }
        var z = new double[n, d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) z[i, j] = (x[i, j] - mu[j]) / sd[j];
        return (z, sd);
    }

    /// <summary>Solve <c>A X = B</c> for matrix RHS <c>B</c> (column by column).</summary>
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
        catch (Exception)
        {
            return null;
        }
        return AllFinite(result) ? result : null;
    }

    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) t[j, i] = a[i, j];
        return t;
    }

    private static double[] MatVec(double[,] m, double[] v)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var r = new double[rows];
        for (int i = 0; i < rows; i++) { double s = 0; for (int j = 0; j < cols; j++) s += m[i, j] * v[j]; r[i] = s; }
        return r;
    }

    private static double[] MatTVec(double[,] m, double[] v)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var r = new double[cols];
        for (int j = 0; j < cols; j++) { double s = 0; for (int i = 0; i < rows; i++) s += m[i, j] * v[i]; r[j] = s; }
        return r;
    }

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static double[] Negate(double[] a)
    {
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = -a[i];
        return r;
    }

    private static int ArgMin(double[] a)
    {
        int idx = 0;
        double best = a[0];
        for (int i = 1; i < a.Length; i++) if (a[i] < best) { best = a[i]; idx = i; }
        return idx;
    }

    private static double Clip(double v, double lo, double hi) => Math.Min(Math.Max(v, lo), hi);
    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
    private static bool AllFinite(double[] a) { foreach (double v in a) if (!IsFinite(v)) return false; return true; }

    private static bool AllFinite(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) if (!IsFinite(a[i, j])) return false;
        return true;
    }
}

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Gp;

/// <summary>
///     A Gaussian-process regressor conditioned on a fixed set of pre-sampled running-trial fantasies —
///     port of <c>ConditionalGPRegressor</c> in <c>optuna/_gp/gp.py</c>, the surrogate the q-batch
///     acquisition functions (<c>qLogEI</c>/<c>qLogPI</c>/<c>qLogEHVI</c>, PORTING_PLAN Phase 3.2 part
///     3e-6) integrate over. It draws standard-normal QMC base samples once
///     (<see cref="Qmc.SampleFromNormalSobol" />, matched to torch ~1e-15), pre-samples fantasy values
///     for the running trials from <c>p(f_Xr | complete trials)</c>, and then draws conditional samples
///     at each query point from <c>p(f_x | f_Xr, complete trials)</c>. Because the base samples are
///     fixed, <see cref="SampleJointPosterior" /> is a <b>deterministic</b> (hence differentiable)
///     function of the query point.
///
///     <para>The conditional posterior of <c>y_x | y_Xr</c> has mean
///     <c>μ_x + cov_xr·cov_rr⁻¹·(y_r − μ_r)</c> and variance <c>cov_xx − cov_xr·cov_rr⁻¹·cov_rx</c> (all
///     <em>posterior</em> covariances). The precomputed <c>cov_rr_post_inv_delta_r</c> and <c>V_r</c>
///     carry the running-conditioning; each query only needs one posterior evaluation plus a small
///     <c>r×r</c> solve.</para>
///
///     <para><b>Tolerance, not bit-exact</b> (the documented GP boundary): the posterior rides the
///     managed triangular solve and the QMC samples match torch to ~1e-15.</para>
/// </summary>
internal sealed class ConditionalGPRegressor
{
    private readonly GPRegressor _gpr;             // the completed-trials-only regressor
    private readonly double[,] _xRunning;          // (r, dim)
    private readonly int _nRunning;
    private readonly int _nQmc;
    private readonly double _stabilizingNoise;

    private readonly double[] _fixedSamplesX;       // (n_qmc,) the query column of the QMC base samples
    private readonly double[,] _covRrPostChol;      // (r, r) Cholesky of cov_rr_post + stab·I
    private readonly double[,] _fantasySamples;     // (n_qmc, r) pre-sampled running fantasies
    private readonly double[,] _covRrPostInvDeltaR; // (r, n_qmc) = cov_rr_post⁻¹ · (fantasy − mean_r)ᵀ
    private readonly double[,] _vR;                 // (N, r) = (cov_Y_Y⁻¹ · cov_fX_fXr)

    internal ConditionalGPRegressor(
        GPRegressor gpr, double[,] xRunning, int nQmcSamples, int qmcSeed, double stabilizingNoise = 1e-12)
    {
        _gpr = gpr;
        _xRunning = xRunning;
        _nRunning = xRunning.GetLength(0);
        _stabilizingNoise = stabilizingNoise;

        // fixed_samples: (n_qmc, r+1) standard-normal Sobol'; the final column is reserved for the query.
        double[,] fixedSamples = Qmc.SampleFromNormalSobol(_nRunning + 1, nQmcSamples, qmcSeed);
        _nQmc = fixedSamples.GetLength(0);
        _fixedSamplesX = new double[_nQmc];
        for (int s = 0; s < _nQmc; s++)
            _fixedSamplesX[s] = fixedSamples[s, _nRunning];

        // Joint posterior at the running points; stabilize the diagonal, then Cholesky.
        (double[] meanR, double[,] covRrPost) = gpr.PosteriorJoint(xRunning);
        for (int i = 0; i < _nRunning; i++)
            covRrPost[i, i] += stabilizingNoise;
        _covRrPostChol = LinAlg.Cholesky(covRrPost);

        // fantasy_samples[s,i] = mean_r[i] + Σ_j fixed_samples[s,j]·chol[i,j]   (= fixed[:, :-1] @ cholᵀ).
        _fantasySamples = new double[_nQmc, _nRunning];
        for (int s = 0; s < _nQmc; s++)
            for (int i = 0; i < _nRunning; i++)
            {
                double acc = meanR[i];
                for (int j = 0; j < _nRunning; j++)
                    acc += fixedSamples[s, j] * _covRrPostChol[i, j];
                _fantasySamples[s, i] = acc;
            }

        // delta_r[i,s] = fantasy[s,i] − mean_r[i];  cov_rr_post_inv_delta_r = cov_rr_post⁻¹ · delta_r.
        var deltaR = new double[_nRunning, _nQmc];
        for (int i = 0; i < _nRunning; i++)
            for (int s = 0; s < _nQmc; s++)
                deltaR[i, s] = _fantasySamples[s, i] - meanR[i];
        _covRrPostInvDeltaR = GPRegressor.SolveCholesky(_covRrPostChol, deltaR, left: true);

        // V_r = (cov_Y_Y⁻¹ · cov_fX_fXr): solve_cholesky(cov_Y_Y_chol, cov_fXr_fX, left=False)ᵀ → (N, r).
        double[,] covFxrFX = gpr.KernelToTrain(xRunning);                                  // (r, N)
        double[,] vrT = GPRegressor.SolveCholesky(gpr.CovYYChol, covFxrFX, left: false);   // (r, N)
        _vR = Transpose(vrT);                                                              // (N, r)
    }

    /// <summary><c>get_fantasy_samples()</c> — the pre-sampled running fantasies, shape <c>(n_qmc, r)</c>.</summary>
    internal double[,] GetFantasySamples() => _fantasySamples;

    /// <summary>Number of QMC samples (the actual drawn count, which may exceed the requested value).</summary>
    internal int NQmc => _nQmc;

    /// <summary>Number of running trials conditioned on.</summary>
    internal int NRunning => _nRunning;

    /// <summary>
    ///     <c>sample_joint_posterior(x, return_fantasy=True)</c> at a single query point — the
    ///     <c>(n_qmc, r+1)</c> joint-posterior samples (the running fantasies followed by the query
    ///     sample per QMC draw).
    /// </summary>
    internal double[,] SampleJointPosterior(double[] x)
        => WithFantasy(Query(x, withGrad: false).Samples);

    /// <summary><c>sample_joint_posterior(x, return_fantasy=False)</c> — just the query samples, shape <c>(n_qmc,)</c>.</summary>
    internal double[] SampleQuerySamples(double[] x) => Query(x, withGrad: false).Samples;

    /// <summary>
    ///     <see cref="SampleJointPosterior" /> plus the gradient of the query-sample column w.r.t. the
    ///     query point (the fantasy columns are constant → zero gradient). Returns <c>YPost (n_qmc, r+1)</c>
    ///     and <c>DSamples (n_qmc, dim)</c> = <c>∂samples[s]/∂x_d</c>.
    /// </summary>
    internal (double[,] YPost, double[,] DSamples) SampleJointPosteriorWithGrad(double[] x)
    {
        (double[] samples, double[,]? dSamples) = Query(x, withGrad: true);
        return (WithFantasy(samples), dSamples!);
    }

    /// <summary>The query samples <c>(n_qmc,)</c> and their gradient <c>(n_qmc, dim)</c> (for <c>qLogEHVI</c>, <c>return_fantasy=False</c>).</summary>
    internal (double[] Samples, double[,] DSamples) SampleQueryWithGrad(double[] x)
    {
        (double[] samples, double[,]? dSamples) = Query(x, withGrad: true);
        return (samples, dSamples!);
    }

    private double[,] WithFantasy(double[] samples)
    {
        var yPost = new double[_nQmc, _nRunning + 1];
        for (int s = 0; s < _nQmc; s++)
        {
            for (int i = 0; i < _nRunning; i++)
                yPost[s, i] = _fantasySamples[s, i];
            yPost[s, _nRunning] = samples[s];
        }
        return yPost;
    }

    /// <summary>
    ///     The core of <c>sample_joint_posterior</c>: the conditional query samples per QMC draw and
    ///     (when <paramref name="withGrad" />) their gradient w.r.t. the query point. Chains the query's
    ///     posterior (<c>μ_x</c>, <c>cov_xx</c>) and the running cross-covariance
    ///     <c>cov_fx_fXr_post = cov_fx_fXr − cov_fx_fX·V_r</c> through the conditional mean/variance.
    /// </summary>
    private (double[] Samples, double[,]? DSamples) Query(double[] x, bool withGrad)
    {
        int dim = x.Length;
        int r = _nRunning;

        double muX, covXx;
        double[]? dMuX = null, dCovXx = null;
        bool covClamped = false;
        if (withGrad)
        {
            (double m, double v, double[] dm, double[] dv, bool clamped) = _gpr.PosteriorWithGrad(x);
            muX = m; covXx = v; dMuX = dm; dCovXx = dv; covClamped = clamped;
        }
        else
        {
            var row = new double[1, dim];
            for (int d = 0; d < dim; d++) row[0, d] = x[d];
            (double[] mean, double[] var) = _gpr.Posterior(row);
            muX = mean[0]; covXx = var[0];
        }
        _ = covClamped;  // the query variance clamp is captured by dCovXx already (0 when clamped).

        (double[] covFxFXr, double[,] dCovFxFXr) = _gpr.KernelRowWithGrad(x, _xRunning);  // (r,), (r, dim)
        (double[] covFxFX, double[,] dCovFxFX) = _gpr.KernelRowToTrainWithGrad(x);        // (N,), (N, dim)
        int nTrain = covFxFX.Length;

        // cov_fx_fXr_post[i] = cov_fx_fXr[i] − Σ_n cov_fx_fX[n]·V_r[n,i].
        var q = new double[r];
        for (int i = 0; i < r; i++)
        {
            double s = 0.0;
            for (int n = 0; n < nTrain; n++)
                s += covFxFX[n] * _vR[n, i];
            q[i] = covFxFXr[i] - s;
        }

        // cond_mean[s] = μ_x + Σ_i q[i]·cov_rr_post_inv_delta_r[i,s].
        var condMean = new double[_nQmc];
        for (int s = 0; s < _nQmc; s++)
        {
            double acc = muX;
            for (int i = 0; i < r; i++)
                acc += q[i] * _covRrPostInvDeltaR[i, s];
            condMean[s] = acc;
        }

        // cond_cov = cov_xx + stab − q·cov_rr_post⁻¹·q (clamped ≥ 0); V = cov_rr_post⁻¹·q.
        double[] vVec = GPRegressor.SolveCholesky(_covRrPostChol, q);
        double quad = 0.0;
        for (int i = 0; i < r; i++)
            quad += vVec[i] * q[i];
        double rawCondCov = covXx + _stabilizingNoise - quad;
        bool condClamped = rawCondCov < 0.0;
        double condCov = condClamped ? 0.0 : rawCondCov;
        double sqrtCondCov = Math.Sqrt(condCov);

        var samples = new double[_nQmc];
        for (int s = 0; s < _nQmc; s++)
            samples[s] = condMean[s] + sqrtCondCov * _fixedSamplesX[s];

        if (!withGrad)
            return (samples, null);

        // dq[i]/dx_d = dcov_fx_fXr[i,d] − Σ_n dcov_fx_fX[n,d]·V_r[n,i].
        var dq = new double[r, dim];
        for (int i = 0; i < r; i++)
            for (int d = 0; d < dim; d++)
            {
                double s = 0.0;
                for (int n = 0; n < nTrain; n++)
                    s += dCovFxFX[n, d] * _vR[n, i];
                dq[i, d] = dCovFxFXr[i, d] - s;
            }

        // d(cond_cov)/dx_d = dcov_xx[d] − 2 Σ_i V[i]·dq[i,d]  (0 when clamped); d√ = dcond_cov/(2√).
        var dSqrtCondCov = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            if (condClamped || sqrtCondCov == 0.0) { dSqrtCondCov[d] = 0.0; continue; }
            double s = 0.0;
            for (int i = 0; i < r; i++)
                s += vVec[i] * dq[i, d];
            double dCondCov = dCovXx![d] - 2.0 * s;
            dSqrtCondCov[d] = dCondCov / (2.0 * sqrtCondCov);
        }

        // d(samples[s])/dx_d = dcond_mean[s,d] + fixed_x[s]·d√cond_cov[d];
        //   dcond_mean[s,d] = dμ_x[d] + Σ_i dq[i,d]·cov_rr_post_inv_delta_r[i,s].
        var dSamples = new double[_nQmc, dim];
        for (int s = 0; s < _nQmc; s++)
            for (int d = 0; d < dim; d++)
            {
                double dcm = dMuX![d];
                for (int i = 0; i < r; i++)
                    dcm += dq[i, d] * _covRrPostInvDeltaR[i, s];
                dSamples[s, d] = dcm + _fixedSamplesX[s] * dSqrtCondCov[d];
            }
        return (samples, dSamples);
    }

    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j, i] = a[i, j];
        return t;
    }
}

namespace OptunaSharp.Gp;

/// <summary>
///     One <see cref="ConditionalGPRegressor" /> per objective, all conditioned on the SAME running
///     trials — port of <c>ListConditionalGPRegressor</c> in <c>optuna/_gp/gp.py</c> (added in Optuna
///     5.0.0 by #6804). It is the shared surrogate of the q-batch multi-objective acquisition functions
///     (<see cref="QLogEHVI" />, <see cref="QLogCEHVI" />): for every QMC draw it holds the per-objective
///     running-trial fantasies and returns the per-objective conditional sample at a candidate point.
/// </summary>
/// <remarks>
///     Objective <c>i</c> draws its QMC base samples with the seed <c>qmcSeed + i</c>, exactly like
///     Optuna, so the fantasies of different objectives are independent but reproducible. Fantasies are
///     pre-sampled once at construction; every later call is a deterministic function of the candidate
///     point, which is what lets the acquisition optimizer use gradients. Callers must treat the returned
///     fantasy arrays as read-only: they are the regressor's cached buffers, shared by every evaluation.
/// </remarks>
internal sealed class ListConditionalGPRegressor
{
    private readonly ConditionalGPRegressor[] _condGprList;
    private readonly double[][,] _fantasySamples;   // per QMC draw: (n_running, n_objectives)

    /// <summary>
    ///     Builds one conditional regressor per objective over the shared running trials and caches
    ///     their fantasies regrouped by QMC draw (Python's <c>torch.stack(..., dim=-1)</c>).
    /// </summary>
    /// <param name="gprList">One Gaussian-process model per objective, in objective order, fitted on completed trials only.</param>
    /// <param name="xRunning">The normalized parameter rows of the running trials, shared by every objective.</param>
    /// <param name="nQmcSamples">The number of fixed quasi-Monte Carlo draws used to approximate posterior expectations.</param>
    /// <param name="qmcSeed">The base QMC seed; objective <c>i</c> uses <c>qmcSeed + i</c>.</param>
    /// <param name="stabilizingNoise">A small variance increment used to avoid singular predictive standard deviations.</param>
    internal ListConditionalGPRegressor(
        GPRegressor[] gprList, double[,] xRunning, int nQmcSamples, int qmcSeed, double stabilizingNoise)
    {
        _condGprList = new ConditionalGPRegressor[gprList.Length];
        for (int i = 0; i < gprList.Length; i++)
            _condGprList[i] = new ConditionalGPRegressor(
                gprList[i], xRunning, nQmcSamples, qmcSeed + i, stabilizingNoise);

        // Regroup (n_objectives × (n_qmc, n_running)) into (n_qmc × (n_running, n_objectives)) once,
        // because every consumer iterates the fantasies per QMC draw (Python: list(fantasy_samples)).
        int nObjectives = gprList.Length;
        int nQmc = _condGprList[0].NQmc;
        int nRunning = _condGprList[0].NRunning;
        _fantasySamples = new double[nQmc][,];
        for (int s = 0; s < nQmc; s++)
        {
            var fantasy = new double[nRunning, nObjectives];
            for (int o = 0; o < nObjectives; o++)
            {
                double[,] perObjective = _condGprList[o].GetFantasySamples();
                for (int r = 0; r < nRunning; r++)
                    fantasy[r, o] = perObjective[s, r];
            }
            _fantasySamples[s] = fantasy;
        }
    }

    /// <summary>The number of QMC draws (the actual drawn count, which may exceed the requested value).</summary>
    internal int NQmc => _fantasySamples.Length;

    /// <summary>The number of objectives, one conditional regressor each.</summary>
    internal int NObjectives => _condGprList.Length;

    /// <summary>
    ///     <c>get_fantasy_samples()</c> — the running-trial fantasies per QMC draw, each a
    ///     <c>(n_running, n_objectives)</c> matrix (Python shape <c>(n_qmc, n_running, n_objectives)</c>).
    /// </summary>
    /// <returns>The cached per-draw fantasy matrices; the arrays are shared buffers and must not be mutated.</returns>
    internal double[][,] GetFantasySamples() => _fantasySamples;

    /// <summary>
    ///     <c>sample_candidate_posterior(x)</c> — the conditional sample of every objective at the
    ///     candidate point for every QMC draw, shape <c>(n_qmc, n_objectives)</c>.
    /// </summary>
    /// <param name="x">The candidate point in normalized search-space coordinates.</param>
    /// <returns>The per-draw, per-objective candidate samples.</returns>
    internal double[,] SampleCandidatePosterior(double[] x)
    {
        var samples = new double[NQmc, NObjectives];
        for (int o = 0; o < NObjectives; o++)
        {
            double[] perObjective = _condGprList[o].SampleQuerySamples(x);
            for (int s = 0; s < perObjective.Length; s++)
                samples[s, o] = perObjective[s];
        }
        return samples;
    }

    /// <summary>
    ///     <see cref="SampleCandidatePosterior" /> plus the gradient of every objective's candidate
    ///     samples with respect to the candidate point — the analytic replacement for autograd through
    ///     <c>sample_joint_posterior(x, return_fantasy=False)</c>.
    /// </summary>
    /// <param name="x">The candidate point in normalized search-space coordinates.</param>
    /// <returns>
    ///     The <c>(n_qmc, n_objectives)</c> samples and, per objective, the <c>(n_qmc, dim)</c> matrix of
    ///     <c>∂sample[s]/∂x_d</c>.
    /// </returns>
    internal (double[,] Samples, double[][,] DSamples) SampleCandidatePosteriorWithGrad(double[] x)
    {
        var samples = new double[NQmc, NObjectives];
        var dSamples = new double[NObjectives][,];
        for (int o = 0; o < NObjectives; o++)
        {
            (double[] perObjective, double[,] dPerObjective) = _condGprList[o].SampleQueryWithGrad(x);
            for (int s = 0; s < perObjective.Length; s++)
                samples[s, o] = perObjective[s];
            dSamples[o] = dPerObjective;
        }
        return (samples, dSamples);
    }
}

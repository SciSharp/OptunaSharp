using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler using Gaussian-process-based Bayesian optimization — port of
///     <c>optuna/samplers/_gp/sampler.py: GPSampler</c>. It fits a Matérn-5/2 ARD Gaussian process to
///     the past trials (kernel hyperparameters by marginal-likelihood maximization with the Gamma /
///     hand-crafted priors) and suggests the next parameters by maximizing an acquisition function
///     with the mixed continuous/discrete optimizer (<see cref="OptimMixed" />).
///
///     <para><b>GP stays native (Tier N):</b> no TorchSharp — the fit rides the hand-derived
///     Matérn-5/2 marginal-likelihood gradient and the managed L-BFGS-B (<see cref="LBfgsB" />), and
///     the acquisition optimization the managed acqf gradient. The path is tolerance-exact vs Optuna
///     (ULP drift from scipy's OpenBLAS BLAS + the erf/ucrt class in LogEI), the documented GP boundary
///     — <b>not</b> bit-for-bit like the MT19937 samplers.</para>
///
///     <para><b>Scope (PORTING_PLAN Phase 3.2 part 3e — complete):</b> every path is ported. Sequential
///     (no running trials): single-objective unconstrained (<see cref="LogEI" />), constrained
///     (<see cref="LogCEI" />), multi-objective (<see cref="LogEHVI" />), and multi-objective constrained
///     (<see cref="LogCEHVI" />). With running trials (asked but not yet told): the q-batch acqfs built
///     on the <see cref="ConditionalGPRegressor" /> — <see cref="QLogEI" />, <see cref="QLogCEI" />,
///     <see cref="QLogEHVI" />, and (since Optuna 5.0.0, #6804) <see cref="QLogCEHVI" /> for the
///     multi-objective constrained case, which earlier versions handled with a Kriging-Believer
///     <see cref="LogCEHVI" />.</para>
/// </summary>
/// <example>
///     <code>
///     var sampler = new GPSampler(seed: 0);
///     var study = Optuna.CreateStudy(sampler: sampler);
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10) - 2, 2), nTrials: 30);
///     </code>
/// </example>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html">Official Optuna GPSampler reference.</seealso>
public class GPSampler : BaseSampler
{
    private const double Eps = 1e-10;

    // Multi-objective and constrained optimization may derive QMC seeds by adding small (~30) offsets
    // to a base seed. Use `1 << 30` as a conservative upper bound. (Unused on the single-objective path.)
    private const int MaxQmcSeedValue = 1 << 30;
    private const string RelativeParamsKey = "gp:relative_params";
    // The value of system_attrs must be less than 2046 characters on RDBStorage.
    private const int SystemAttrMaxLength = 2045;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers._gp.sampler");

    private readonly LazyRandomState _rng;
    private readonly BaseSampler _independentSampler;
    private readonly IntersectionSearchSpace _intersectionSearchSpace;
    private readonly int _nStartupTrials;
    private readonly double _minimumNoise;
    private readonly bool _deterministic;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private bool _warnIndependentSampling;

    // We cache the kernel parameters for initial values of fitting the next time.
    private List<GPRegressor>? _gprsCacheList;
    private List<GPRegressor>? _constraintsGprsCacheList;

    // Control parameters of the acquisition function optimization.
    private readonly int _nPreliminarySamples = 2048;
    private readonly int _nLocalSearch = 10;
    private readonly double _tol = 1e-4;
    private readonly int _nQmcSamplesQei = 128;   // QMC samples for the q-batch expected-improvement estimate
    private readonly int _nQmcSamplesEhvi = 128;  // BoTorch default for the EHVI QMC estimate

    /// <param name="seed">Random seed to initialize the internal RNG. Defaults to <c>null</c> (picked randomly).</param>
    /// <param name="independentSampler">
    ///     Sampler used for initial sampling (the first <paramref name="nStartupTrials" /> trials) and
    ///     for conditional parameters. Defaults to a <see cref="RandomSampler" /> with the same seed.
    /// </param>
    /// <param name="nStartupTrials">Number of initial trials sampled independently. Defaults to 10.</param>
    /// <param name="deterministicObjective">
    ///     Whether the objective is deterministic. If <c>true</c>, the noise variance is fixed at the
    ///     minimum. Defaults to <c>false</c> (experimental).
    /// </param>
    /// <param name="constraintsFunc">
    ///     Optional constraints function evaluated after each successful trial. <b>Deprecated in
    ///     v5.0.0</b> — use <c>Trial.SetConstraint</c> instead.
    /// </param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <summary>Configures Gaussian-process fitting and the independent sampler used before enough observations exist.</summary>
    public GPSampler(
        int? seed = null,
        BaseSampler? independentSampler = null,
        int nStartupTrials = 10,
        bool deterministicObjective = false,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        bool warnIndependentSampling = true)
    {
        _rng = new LazyRandomState(seed);
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _intersectionSearchSpace = new IntersectionSearchSpace();
        _nStartupTrials = nStartupTrials;
        _minimumNoise = Prior.DefaultMinimumNoiseVar;
        _deterministic = deterministicObjective;
        _constraintsFunc = constraintsFunc;
        _warnIndependentSampling = warnIndependentSampling;

        if (constraintsFunc != null)
            OptunaWarnings.Warn(
                string.Format(Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0")
                + " Use `optuna.trial.Trial.set_constraint` instead.",
                WarningCategory.FutureWarning);
        // Optuna raises an ExperimentalWarning (warn_experimental_argument) for deterministic_objective.
        // OptunaSharp targets the Optuna 5.x surface and treats it as stable, so it is omitted.
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed(), fresh OS entropy: 128 bits through numpy's own key construction,
        // so concurrent workers never share a stream (THREADING.md §6.3).
        OsEntropySeed.Reseed(_rng.Rng);
        _independentSampler.ReseedRng();
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _intersectionSearchSpace.Calculate(study))
        {
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var states = new[] { TrialState.Complete, TrialState.Running };
        var trials = study.GetTrialsInternal(deepcopy: false, states: states, useCache: false);
        var completedTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        var runningTrials = trials
            .Where(t => t.State == TrialState.Running
                        && searchSpace.Keys.All(k => GetParams(t).ContainsKey(k)))
            .ToList();

        if (completedTrials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        var sampled = SampleRelativeImpl(study, completedTrials, runningTrials, searchSpace);

        if (sampled.Count > 0)
        {
            // Share the params obtained by the relative sampling with the other processes.
            // OrderedDict preserves the search-space key order, so this matches json.dumps(params).
            string paramsStr = PyJson.Dumps(sampled);
            for (int i = 0; i < paramsStr.Length; i += SystemAttrMaxLength)
            {
                int len = Math.Min(SystemAttrMaxLength, paramsStr.Length - i);
                study.Storage.SetTrialSystemAttr(
                    trial.TrialId, $"{RelativeParamsKey}:{i / SystemAttrMaxLength}",
                    paramsStr.Substring(i, len));
            }
        }
        return sampled;
    }

    /// <summary>Fits objective and constraint Gaussian processes and optimizes the acquisition function.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <param name="runningTrials">Running trial snapshots used to account for pending evaluations.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <returns>External parameter values selected by the acquisition optimizer.</returns>
    private OrderedDict<string, object?> SampleRelativeImpl(
        Study study, List<FrozenTrial> completedTrials, List<FrozenTrial> runningTrials,
        OrderedDict<string, BaseDistribution> searchSpace)
    {
        var internalSearchSpace = new GpSearchSpace(searchSpace);
        var completedParams = completedTrials
            .Select(t => t.Params)
            .ToList();
        double[,] normalizedParams = internalSearchSpace.GetNormalizedParams(completedParams);

        // The running trials' normalized params (via the shared gp:relative_params attrs) — the
        // q-batch acquisition functions condition on the trials already in flight.
        bool hasRunning = runningTrials.Count > 0;
        double[,]? normalizedParamsOfRunningTrials = hasRunning
            ? internalSearchSpace.GetNormalizedParams(runningTrials.Select(GetParams).ToList())
            : null;

        var directions = study.Directions;
        int nObjectives = directions.Count;
        var sign = directions.Select(d => d == StudyDirection.Minimize ? -1.0 : 1.0).ToArray();

        int n = completedTrials.Count;
        var values = new double[n, nObjectives];
        for (int t = 0; t < n; t++)
        {
            IReadOnlyList<double> tv = completedTrials[t].Values!;
            for (int j = 0; j < nObjectives; j++)
                values[t, j] = sign[j] * tv[j];
        }
        (double[,] standardizedScoreVals, _, _) = StandardizeValues(values);

        if (_gprsCacheList != null
            && _gprsCacheList[0].InverseSquaredLengthscales.Length != internalSearchSpace.Dim)
            _gprsCacheList = null;  // Clear cache if the search space changes.

        // Fit one GP regressor per objective over the completed trials.
        bool[] isCategorical = internalSearchSpace.IsCategorical;
        var gprsList = new List<GPRegressor>();
        for (int i = 0; i < nObjectives; i++)
        {
            var yi = new double[n];
            for (int t = 0; t < n; t++)
                yi[t] = standardizedScoreVals[t, i];
            GPRegressor? cache = _gprsCacheList != null ? _gprsCacheList[i] : null;
            gprsList.Add(GPRegressor.FitKernelParams(
                normalizedParams, yi, isCategorical, _minimumNoise, _deterministic, cache));
        }
        _gprsCacheList = gprsList;

        IAcquisitionFunc acqf;
        double[,]? bestParams;
        if (!ConstrainedOptimization.IsConstrainedOptimization(completedTrials))
        {
            if (nObjectives == 1)
            {
                double threshold = ColumnMax(standardizedScoreVals, 0);
                if (hasRunning)
                {
                    int qmcSeed = (int)NumPyDraws.RandInt(_rng.Rng, 0, MaxQmcSeedValue);
                    acqf = new QLogEI(
                        gprsList[0], internalSearchSpace, threshold, _nQmcSamplesQei, qmcSeed,
                        normalizedParamsOfRunningTrials!);
                }
                else
                    acqf = new LogEI(gprsList[0], internalSearchSpace, threshold);
                bestParams = BestParamsRow(normalizedParams, ArgMaxColumn(standardizedScoreVals, 0));
            }
            else
            {
                int qmcSeed = (int)NumPyDraws.RandInt(_rng.Rng, 0, MaxQmcSeedValue);
                if (hasRunning)
                    acqf = new QLogEHVI(
                        gprsList.ToArray(), internalSearchSpace, standardizedScoreVals, _nQmcSamplesEhvi,
                        qmcSeed, normalizedParamsOfRunningTrials!);
                else
                    acqf = new LogEHVI(
                        gprsList.ToArray(), internalSearchSpace, standardizedScoreVals, _nQmcSamplesEhvi, qmcSeed);
                bestParams = GetBestParamsForMultiObjective(normalizedParams, standardizedScoreVals);
            }
        }
        else
        {
            if (nObjectives == 1)
            {
                (double[,] constraintVals, bool[] isFeasible) =
                    GetConstraintValsAndFeasibility(completedTrials);
                // y_with_neginf = np.where(is_feasible, standardized_score_vals[:, 0], -inf).
                var yWithNegInf = new double[n];
                for (int t = 0; t < n; t++)
                    yWithNegInf[t] = isFeasible[t] ? standardizedScoreVals[t, 0] : double.NegativeInfinity;
                (List<GPRegressor> constrGprList, double[] constrThresholdList) =
                    GetConstraintsAcqfArgs(constraintVals, internalSearchSpace, normalizedParams);
                int iOpt = ArgMax(yWithNegInf);
                double bestFeasibleY = yWithNegInf[iOpt];
                if (hasRunning)
                {
                    int qmcSeed = (int)NumPyDraws.RandInt(_rng.Rng, 0, MaxQmcSeedValue);
                    acqf = new QLogCEI(
                        gprsList[0], internalSearchSpace, bestFeasibleY, _nQmcSamplesQei, qmcSeed,
                        constrGprList, constrThresholdList, normalizedParamsOfRunningTrials!);
                }
                else
                    acqf = new LogCEI(
                        gprsList[0], internalSearchSpace, bestFeasibleY, constrGprList, constrThresholdList);
                bestParams = double.IsNegativeInfinity(bestFeasibleY)
                    ? null
                    : BestParamsRow(normalizedParams, iOpt);
            }
            else
            {
                (double[,] constraintVals, bool[] isFeasible) =
                    GetConstraintValsAndFeasibility(completedTrials);
                (List<GPRegressor> constrGprList, double[] constrThresholdList) =
                    GetConstraintsAcqfArgs(constraintVals, internalSearchSpace, normalizedParams);
                bool isAllInfeasible = !isFeasible.Any(f => f);
                // Optuna 5.0.0 (#6804): with running trials the MO-constrained case is the q-batch
                // qLogCEHVI (fantasy-conditioned); without them it stays the sequential LogCEHVI. Both
                // branches draw exactly one QMC seed, so the sampler's RNG stream is unchanged.
                int qmcSeed = (int)NumPyDraws.RandInt(_rng.Rng, 0, MaxQmcSeedValue);
                double[,]? yFeasible = isAllInfeasible ? null : FilterRows(standardizedScoreVals, isFeasible);
                acqf = hasRunning
                    ? new QLogCEHVI(
                        gprsList.ToArray(), internalSearchSpace, yFeasible, _nQmcSamplesEhvi, qmcSeed,
                        constrGprList, constrThresholdList, normalizedParamsOfRunningTrials!)
                    : new LogCEHVI(
                        gprsList.ToArray(), internalSearchSpace, yFeasible, _nQmcSamplesEhvi, qmcSeed,
                        constrGprList, constrThresholdList);
                bestParams = isAllInfeasible
                    ? null
                    : GetBestParamsForMultiObjective(
                        FilterRows(normalizedParams, isFeasible), FilterRows(standardizedScoreVals, isFeasible));
            }
        }

        (double[] normalizedParam, _) = OptimMixed.OptimizeAcqfMixed(
            acqf, bestParams, _nPreliminarySamples, _nLocalSearch, _tol, _rng.Rng);

        Dictionary<string, object?> unnormalized = internalSearchSpace.GetUnnormalizedParam(normalizedParam);
        var result = new OrderedDict<string, object?>();
        foreach (var name in searchSpace.Keys)
            result[name] = unnormalized[name];
        return result;
    }

    /// <summary>
    ///     <c>_get_constraint_vals_and_feasibility</c> — stack each trial's constraint values into an
    ///     <c>(n, n_constraints)</c> matrix (raising the verbatim <c>ValueError</c> when the count
    ///     differs across trials) and the per-trial feasibility mask (<c>all constraints ≤ 0</c>).
    /// </summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <returns>Constraint rows and a feasibility mask aligned with the supplied trials.</returns>
    /// <exception cref="NumSharp.ValueError">Trials expose different numbers of constraints.</exception>
    private static (double[,] ConstraintVals, bool[] IsFeasible) GetConstraintValsAndFeasibility(
        List<FrozenTrial> trials)
    {
        var rows = trials.Select(t => t.Constraints.Values.ToList()).ToList();
        int nc = rows[0].Count;
        if (rows.Any(r => r.Count != nc))
            throw new NumSharp.ValueError("The number of constraints must be the same for all trials.");

        int n = trials.Count;
        var constraintVals = new double[n, nc];
        var isFeasible = new bool[n];
        for (int i = 0; i < n; i++)
        {
            bool feasible = true;
            for (int j = 0; j < nc; j++)
            {
                constraintVals[i, j] = rows[i][j];
                if (!(rows[i][j] <= 0.0))
                    feasible = false;
            }
            isFeasible[i] = feasible;
        }
        return (constraintVals, isFeasible);
    }

    /// <summary>
    ///     <c>_get_constraints_acqf_args</c> — fit one GP per constraint (over the sign-flipped,
    ///     standardized constraint values, since constraints are minimized) and return the constraint
    ///     GPs together with their standardized-zero thresholds <c>−mean / max(EPS, std)</c>. Uses the
    ///     cross-trial <see cref="_constraintsGprsCacheList" /> for warm starts, cleared when the
    ///     search-space dimension changes (mirroring Optuna's check against the objective cache).
    /// </summary>
    /// <param name="constraintVals">Constraint observations aligned with completed trial rows.</param>
    /// <param name="internalSearchSpace">Encoded numerical search-space bounds and distribution metadata.</param>
    /// <param name="normalizedParams">Trial coordinates normalized to the unit numerical box.</param>
    /// <returns>Fitted constraint regressors and their standardized zero-feasibility thresholds.</returns>
    private (List<GPRegressor> Gprs, double[] Thresholds) GetConstraintsAcqfArgs(
        double[,] constraintVals, GpSearchSpace internalSearchSpace, double[,] normalizedParams)
    {
        int n = constraintVals.GetLength(0);
        int nc = constraintVals.GetLength(1);
        // Flip the sign of constraints since they are always to be minimized.
        var negConstraintVals = new double[n, nc];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nc; j++)
                negConstraintVals[i, j] = -constraintVals[i, j];
        (double[,] standardized, double[] means, double[] stds) = StandardizeValues(negConstraintVals);

        if (_gprsCacheList != null
            && _gprsCacheList[0].InverseSquaredLengthscales.Length != internalSearchSpace.Dim)
            _constraintsGprsCacheList = null;  // Clear cache if the search space changes.

        var thresholds = new double[nc];
        for (int j = 0; j < nc; j++)
            thresholds[j] = -means[j] / Math.Max(Eps, stds[j]);

        bool[] isCategorical = internalSearchSpace.IsCategorical;
        var gprs = new List<GPRegressor>();
        for (int j = 0; j < nc; j++)
        {
            var vals = new double[n];
            for (int i = 0; i < n; i++)
                vals[i] = standardized[i, j];
            GPRegressor? cache = _constraintsGprsCacheList != null ? _constraintsGprsCacheList[j] : null;
            gprs.Add(GPRegressor.FitKernelParams(
                normalizedParams, vals, isCategorical, _minimumNoise, _deterministic, cache));
        }
        _constraintsGprsCacheList = gprs;
        return (gprs, thresholds);
    }

    /// <summary>NumPy <c>np.argmax</c>: the first index of the maximum (−inf-safe for all-infeasible masks).</summary>
    /// <param name="values">Scores compared to choose the first maximum.</param>
    /// <returns>The first maximum index.</returns>
    private static int ArgMax(double[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>First argmax of column <paramref name="col" /> — <c>np.argmax(standardized_score_vals)</c> for a single objective.</summary>
    /// <param name="mat">Matrix whose specified row participates in the operation.</param>
    /// <param name="col">Zero-based column index or column-shaped matrix.</param>
    /// <returns>The first row attaining the largest value in the selected column.</returns>
    private static int ArgMaxColumn(double[,] mat, int col)
    {
        int best = 0;
        for (int i = 1; i < mat.GetLength(0); i++)
            if (mat[i, col] > mat[best, col]) best = i;
        return best;
    }

    /// <summary>Max of column <paramref name="col" /> — <c>standardized_score_vals[:, col].max()</c>.</summary>
    /// <param name="mat">Matrix whose specified row participates in the operation.</param>
    /// <param name="col">Zero-based column index or column-shaped matrix.</param>
    /// <returns>The maximum selected-column value, or negative infinity for an empty column.</returns>
    private static double ColumnMax(double[,] mat, int col)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < mat.GetLength(0); i++)
            if (mat[i, col] > max) max = mat[i, col];
        return max;
    }

    /// <summary><c>normalized_params[idx, np.newaxis]</c> — the single best point as a <c>(1, dim)</c> warmstart.</summary>
    /// <param name="normalizedParams">Trial coordinates normalized to the unit numerical box.</param>
    /// <param name="idx">Zero-based indices selecting vector entries or population rows.</param>
    /// <returns>The selected normalized point as a one-row warm-start matrix.</returns>
    private static double[,] BestParamsRow(double[,] normalizedParams, int idx)
    {
        int dim = normalizedParams.GetLength(1);
        var row = new double[1, dim];
        for (int d = 0; d < dim; d++)
            row[0, d] = normalizedParams[idx, d];
        return row;
    }

    /// <summary>
    ///     <c>_get_best_params_for_multi_objective</c> — the Pareto-front warmstart points for the MO
    ///     acqf optimizer: pick up to <c>n_local_search // 2</c> of the current Pareto-optimal normalized
    ///     params uniformly at random (<c>rng.choice(n_pareto, size, replace=False)</c> ==
    ///     <c>permutation(n_pareto)[:size]</c>, bit-exact via the verified index shuffle).
    /// </summary>
    /// <param name="normalizedParams">Trial coordinates normalized to the unit numerical box.</param>
    /// <param name="standardizedScoreVals">Completed objective observations after direction conversion and standardization.</param>
    /// <returns>A random subset of Pareto-optimal normalized points used to initialize acquisition search.</returns>
    private double[,] GetBestParamsForMultiObjective(double[,] normalizedParams, double[,] standardizedScoreVals)
    {
        int n = standardizedScoreVals.GetLength(0);
        int m = standardizedScoreVals.GetLength(1);
        var negStd = new double[n][];
        for (int r = 0; r < n; r++)
        {
            negStd[r] = new double[m];
            for (int j = 0; j < m; j++)
                negStd[r][j] = -standardizedScoreVals[r, j];
        }
        bool[] onFront = MultiObjective.IsParetoFront(negStd, assumeUniqueLexsorted: false);
        var paretoIdx = new List<int>();
        for (int r = 0; r < n; r++)
            if (onFront[r]) paretoIdx.Add(r);
        int nPareto = paretoIdx.Count;
        int size = Math.Min(_nLocalSearch / 2, nPareto);

        // rng.choice(n_pareto, size, replace=False) = permutation(n_pareto)[:size]. NumPy's
        // permutation is arange + Fisher-Yates shuffle (bit-exact via _rng.Rng.shuffle — the GridSampler
        // pattern), so the RNG advances by the full permutation regardless of size.
        NDArray permArr = np.arange(nPareto);
        _rng.Rng.shuffle(permArr);
        long[] perm = permArr.ToArray<long>();

        int dim = normalizedParams.GetLength(1);
        var best = new double[size, dim];
        for (int r = 0; r < size; r++)
        {
            int idx = paretoIdx[(int)perm[r]];
            for (int d = 0; d < dim; d++)
                best[r, d] = normalizedParams[idx, d];
        }
        return best;
    }

    /// <summary><c>a[mask]</c> for a 2-D array — the rows where <paramref name="mask" /> is <c>true</c>, in order.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="mask">Boolean selection mask aligned with population rows.</param>
    /// <returns>A detached matrix containing selected rows in source order.</returns>
    private static double[,] FilterRows(double[,] a, bool[] mask)
    {
        int cols = a.GetLength(1);
        int kept = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) kept++;
        var result = new double[kept, cols];
        int r = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (!mask[i]) continue;
            for (int c = 0; c < cols; c++)
                result[r, c] = a[i, c];
            r++;
        }
        return result;
    }

    /// <summary>
    ///     <c>_standardize_values</c> — clip non-finite values (<c>warn_and_convert_inf</c>) then
    ///     per-objective standardize <c>(v − mean) / max(EPS, std)</c>, with the column means and population stds taken
    ///     by NumSharp's <c>np.mean(values, axis=0)</c> / <c>np.std(values, axis=0)</c> over the whole
    ///     <c>(n_trials, n_objectives)</c> matrix, NumPy's reduction schedule included. Returns the standardized matrix plus
    ///     the per-column means/stds (the constraint path needs them for its thresholds).
    /// </summary>
    /// <param name="values">Objective observations with one trial per row and one objective per column.</param>
    /// <returns>Standardized objective columns, their original means, and standard deviations.</returns>
    /// <remarks>
    ///     <para>
    ///         NumPy reduces axis 0 of a C-contiguous matrix by adding whole rows in order, a sequential sum per column;
    ///         a single column, shape <c>(n, 1)</c>, collapses to one contiguous run and is summed pairwise instead. The
    ///         former per-column <see cref="MathUtil.Mean" />/<see cref="MathUtil.Std" /> summed every column pairwise, so
    ///         a multi-objective study with eight or more trials standardized differently from Optuna in the last bits
    ///         (measured against NumPy 2.4.2: 17 of 45 two- and three-objective column statistics equal, all 9
    ///         single-objective ones equal); NumSharp follows NumPy's schedule in both cases (54 of 54).
    ///     </para>
    ///     <para>The matrix is viewed zero-copy and every transient array is disposed before returning.</para>
    /// </remarks>
    private static (double[,] Standardized, double[] Means, double[] Stds) StandardizeValues(double[,] values)
    {
        double[,] clipped = WarnAndConvertInf(values);
        int n = clipped.GetLength(0);
        int m = clipped.GetLength(1);
        double[] means, stds;
        using (NDArray view = np.array(clipped, copy: false))
        using (NDArray meanArray = np.mean(view, axis: 0))
        using (NDArray stdArray = np.std(view, axis: 0))
        {
            means = meanArray.ToArray<double>();
            stds = stdArray.ToArray<double>();
        }
        // (clipped_values - means) / np.maximum(EPS, stds): one subtraction and one division per element, as NumPy's
        // broadcast ufuncs compute them.
        var result = new double[n, m];
        for (int j = 0; j < m; j++)
        {
            double denom = Math.Max(Eps, stds[j]);
            for (int t = 0; t < n; t++)
                result[t, j] = (clipped[t, j] - means[j]) / denom;
        }
        return (result, means, stds);
    }

    /// <summary>
    ///     <c>warn_and_convert_inf</c> — clip each column's non-finite entries to that column's finite
    ///     min/max (or 0 when a whole column is non-finite). Fast-paths the all-finite case (the common
    ///     one) to return the input unchanged.
    /// </summary>
    /// <param name="values">Objective observations with one trial per row and one objective per column.</param>
    /// <returns>Objective values with infinities replaced by finite values for Gaussian-process fitting.</returns>
    private static double[,] WarnAndConvertInf(double[,] values)
    {
        int n = values.GetLength(0);
        int m = values.GetLength(1);
        bool allFinite = true;
        for (int i = 0; i < n && allFinite; i++)
            for (int j = 0; j < m; j++)
                if (!double.IsFinite(values[i, j])) { allFinite = false; break; }
        if (allFinite)
            return values;

        OptunaWarnings.Warn(
            "Clip non-finite values to the min/max finite values for GP fittings.",
            WarningCategory.UserWarning);
        var result = new double[n, m];
        for (int j = 0; j < m; j++)
        {
            double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
            bool anyFinite = false;
            for (int i = 0; i < n; i++)
                if (double.IsFinite(values[i, j]))
                {
                    anyFinite = true;
                    lo = Math.Min(lo, values[i, j]);
                    hi = Math.Max(hi, values[i, j]);
                }
            if (!anyFinite) { lo = 0.0; hi = 0.0; }
            for (int i = 0; i < n; i++)
            {
                double v = values[i, j];
                result[i, j] = double.IsFinite(v) ? v : Math.Min(Math.Max(v, lo), hi);
            }
        }
        return result;
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        if (_warnIndependentSampling)
        {
            var completedTrials = study.GetTrialsInternal(
                deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);
            if (completedTrials.Count >= _nStartupTrials)
                LogIndependentSampling(trial, paramName);
        }
        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <summary>Explains why the current parameter is delegated to the independent sampler.</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="paramName">Name of the parameter being suggested or looked up.</param>
    private void LogIndependentSampling(FrozenTrial trial, string paramName)
        => Logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "dynamic search space is not supported by GPSampler"));

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    /// <exception cref="System.Exception">A configured after-trial or constraint callback fails; its exception propagates.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        if (_constraintsFunc != null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
        _independentSampler.AfterTrial(study, trial, state, values);
    }

    /// <summary>
    ///     <c>_get_params</c> — a finished trial's params, or (for a running trial) the params it shared
    ///     via the chunked <c>gp:relative_params:{i}</c> system attrs, merged with any concrete params.
    /// </summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <returns>The adaptation coefficients for the current covariance rank.</returns>
    private static IReadOnlyDictionary<string, object?> GetParams(FrozenTrial trial)
    {
        // Read-only throughout: a finished record may be compact (plan 13, item S12), and the mutable accessor would
        // give it its dictionary back on every ask.
        if (trial.State.IsFinished())
            return trial.Params;

        var parts = new List<string>();
        int i = 0;
        while (trial.SystemAttrs.TryGetValue($"{RelativeParamsKey}:{i}", out object? v)
               && v is string s && s.Length > 0)
        {
            parts.Add(s);
            i++;
        }
        if (parts.Count == 0)
            return trial.Params;

        var merged = ParseJsonObject(string.Concat(parts));
        foreach (var kv in trial.Params)
            merged[kv.Key] = kv.Value;
        return merged;
    }

    /// <summary>Converts stored JSON object members into the runtime values used by sampler caches.</summary>
    /// <param name="json">JSON text containing stored sampler attributes.</param>
    /// <returns>Runtime values keyed by JSON property name.</returns>
    /// <exception cref="System.Text.Json.JsonException">The stored attribute is not valid JSON.</exception>
    private static Dictionary<string, object?> ParseJsonObject(string json)
    {
        var dict = new Dictionary<string, object?>();
        using var doc = JsonDocument.Parse(json);
        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
            dict[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.Number => prop.Value.TryGetInt64(out long l) ? l : prop.Value.GetDouble(),
                JsonValueKind.String => prop.Value.GetString(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null,
            };
        return dict;
    }
}

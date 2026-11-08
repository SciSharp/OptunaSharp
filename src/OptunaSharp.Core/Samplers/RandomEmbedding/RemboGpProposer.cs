// REMBO's paper-faithful proposal engine (Wang, Hutter, Zoghi, Matheson & de Freitas, JAIR 55 (2016)):
// the inner Bayesian optimizer of Algorithm 2 with the Algorithm 3 adaptive length-scale scheme (§3.3).
// It replaces the previous default (an inner GPSampler over a shadow study, Matern-5/2 ARD) with exactly
// the paper's method:
//
//   * surrogate  = the isotropic squared-exponential GP with a single length scale (SquaredExponentialGp,
//                  Def. 7), plus the §3.2 categorical Hamming kernel when the high-dimensional kernel
//                  carries categorical parameters;
//   * acquisition = expected improvement, maximized over the box Y = [−√d, √d]^d by running BOTH DIRECT
//                  and CMA-ES and taking the best (GlobalMinimizer, §3.3);
//   * hyperparameter fit = the length scale ℓ (and, with categoricals, the Hamming scale λ) chosen by
//                  maximizing the marginal likelihood over [L, U] with DIRECT + CMA-ES, under Algorithm 3:
//                  the upper bound U is shrunk to max(0.9·ℓ, L) after 5 consecutive "exploiting" steps
//                  (posterior std at the chosen point below t_σ = 0.002), and ℓ is re-fit every 20 steps or
//                  whenever U shrinks. Bounds [L, U] = [0.01, 50] and t_σ = 0.002 are the paper's §3.3 values.
//
// The two REMBO kernels (§3.2) differ only in the space the GP measures similarity in: the low-dimensional
// kernel models the objective over the latent point y directly; the high-dimensional kernel models it over
// the projected real parameters p_X(A·y) (so it also embeds categoricals). Both maximize the acquisition
// over y ∈ Y. Algorithm 3 state is kept in memory per interleaved embedding and advanced as trials
// complete; on a cold restart from storage it re-initializes (U = 50) and re-fits from the stored history —
// the paper has no notion of resumption, so this is a documented, run-contiguous fidelity.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.ModelBased.Gp;

namespace OptunaSharp.Samplers.RandomEmbedding;

/// <summary>
///     The paper-faithful REMBO inner optimizer: a squared-exponential GP with expected improvement
///     maximized by DIRECT + CMA-ES and the Algorithm-3 adaptive length scale (Wang et al. 2016, §3.2-3.3).
///     One instance serves all interleaved embeddings; each embedding (run) keeps its own Algorithm-3 state
///     and its history is supplied by the sampler.
/// </summary>
internal sealed class RemboGpProposer
{
    // Algorithm 3 constants (Wang et al. 2016, §3.3).
    private const double LengthScaleLower = 0.01;   // L
    private const double LengthScaleUpperInit = 50.0;   // U (initial)
    private const double CategoricalLower = 0.01;   // Hamming-scale bounds (paper does not adapt λ; fit within a fixed box)
    private const double CategoricalUpper = 50.0;
    private const double ExploitThreshold = 0.002;  // t_σ: posterior std below this ⇒ the step is "exploiting"
    private const int ExploitStreakToShrink = 5;    // C: consecutive exploiting steps that trigger a U shrink
    private const double ShrinkFactor = 0.9;        // U ← max(0.9·ℓ, L)
    private const int RefitPeriod = 20;             // re-fit ℓ every 20 steps
    private const double AcqStabilizingNoise = 1e-12;   // added to posterior variance so σ is never exactly 0 in EI

    private readonly RandomEmbeddingProjection _embedding;
    private readonly bool _highDimensional;
    private readonly int _nStartupTrials;
    private readonly uint _baseSeed;
    private readonly int _acqDirectEvals;
    private readonly int _acqCmaEvals;
    private readonly int _hyperDirectEvals;
    private readonly int _hyperCmaEvals;

    // Per-embedding (run) Algorithm-3 state, advanced as the sampler proposes new points for that embedding.
    private readonly Dictionary<int, RunState> _runs = new();

    /// <summary>The mutable Algorithm-3 state of one interleaved embedding.</summary>
    private sealed class RunState
    {
        /// <summary>The current length-scale upper bound U (shrinks under Algorithm 3); starts at 50.</summary>
        internal double UpperBound = LengthScaleUpperInit;
        /// <summary>The consecutive-exploit counter C (resets to 0 on a non-exploiting step or after a shrink).</summary>
        internal int ExploitStreak;
        /// <summary>Steps taken since the last hyperparameter re-fit (initialized high to force a first fit).</summary>
        internal int StepsSinceRefit = int.MaxValue;
        /// <summary>The current fitted length scale ℓ (valid once <see cref="Fitted" /> is true).</summary>
        internal double LengthScale = 1.0;
        /// <summary>The current fitted categorical Hamming scale λ (only meaningful with categorical coordinates).</summary>
        internal double CategoricalScale = 1.0;
        /// <summary>Whether ℓ has been fit at least once.</summary>
        internal bool Fitted;
        /// <summary>The number of post-startup Bayesian-optimization steps proposed for this embedding.</summary>
        internal int StepCount;
    }

    /// <summary>Configures the proposer.</summary>
    /// <param name="embedding">The shared random embedding (supplies the box <c>Y</c>, the dimension, and the projection).</param>
    /// <param name="highDimensional"><c>true</c> for the §3.2 high-dimensional kernel (GP over the projected real parameters);
    ///     <c>false</c> for the low-dimensional kernel (GP over the latent point <c>y</c>).</param>
    /// <param name="nStartupTrials">Uniform-random latent points drawn per embedding before its GP engages.</param>
    /// <param name="baseSeed">Seed anchoring the (deterministic, dimension-independent) per-step optimizer streams.</param>
    /// <param name="acqDirectEvals">DIRECT budget for the acquisition maximization.</param>
    /// <param name="acqCmaEvals">CMA-ES budget for the acquisition maximization.</param>
    /// <param name="hyperDirectEvals">DIRECT budget for the length-scale marginal-likelihood fit.</param>
    /// <param name="hyperCmaEvals">CMA-ES budget for the length-scale marginal-likelihood fit.</param>
    internal RemboGpProposer(
        RandomEmbeddingProjection embedding, bool highDimensional, int nStartupTrials, uint baseSeed,
        int acqDirectEvals = 1500, int acqCmaEvals = 1500, int hyperDirectEvals = 200, int hyperCmaEvals = 200)
    {
        _embedding = embedding;
        _highDimensional = highDimensional;
        _nStartupTrials = nStartupTrials;
        _baseSeed = baseSeed;
        _acqDirectEvals = acqDirectEvals;
        _acqCmaEvals = acqCmaEvals;
        _hyperDirectEvals = hyperDirectEvals;
        _hyperCmaEvals = hyperCmaEvals;
    }

    /// <summary>
    ///     Proposes the next latent point for one embedding. During startup it draws uniformly from the box
    ///     <c>Y = [−√d, √d]^d</c> (a deterministic, dimension-independent draw); afterwards it fits the
    ///     squared-exponential GP on this embedding's history (re-fitting the length scale under Algorithm 3)
    ///     and returns the <c>y ∈ Y</c> whose kernel-space image maximizes expected improvement.
    /// </summary>
    /// <param name="run">The embedding (run) index whose Algorithm-3 state and history are used.</param>
    /// <param name="latentHistory">The latent points <c>y_i</c> of this embedding's completed trials, in order (each length <c>d</c>).</param>
    /// <param name="maxValues">The matching maximization-oriented objective values (already sign-flipped for a minimizing study).</param>
    /// <param name="embeddedSpace">The embedded real search space (used only by the high-dimensional kernel to project and normalize).</param>
    /// <param name="pendingLatents">
    ///     The latent points of this embedding's trials that are still running (proposed, not yet evaluated). They
    ///     join the acquisition's GP as observations at the worst value seen (a pessimistic constant liar), so a
    ///     proposal made while they run is not their point again. Empty in a sequential run, which is then unchanged.
    /// </param>
    /// <returns>The proposed latent point <c>y ∈ Y</c> (length <c>d</c>).</returns>
    /// <remarks>
    ///     Without the liar, every proposal made before a trial finished maximized the same acquisition on the same
    ///     history, and DIRECT is deterministic, so concurrent asks received one point: 21 of 64 trials at 8 workers.
    ///     The length-scale fit (Algorithm 3) still sees real observations only; the liar shapes the acquisition alone.
    /// </remarks>
    internal double[] Propose(
        int run,
        IReadOnlyList<double[]> latentHistory,
        IReadOnlyList<double> maxValues,
        OrderedDict<string, BaseDistribution> embeddedSpace,
        IReadOnlyList<double[]> pendingLatents)
    {
        int d = _embedding.Dim;
        RunState state = GetRunState(run);

        // Startup, or nothing to model yet (the high-dimensional kernel also needs an embedded space): draw a
        // uniform latent point. Seeded by (baseSeed, run, stepCount) so it is deterministic and independent
        // of the extrinsic dimension D (which keeps Theorem 4 / D-independence an exact identity).
        int n = latentHistory.Count;
        bool canModel = n >= _nStartupTrials && (!_highDimensional || embeddedSpace.Count > 0);
        if (!canModel)
        {
            double[] yRand = RandomY(run, state.StepCount);
            state.StepCount++;
            return yRand;
        }

        // Build the GP's training inputs in its kernel space: the latent y (low-dim) or the normalized
        // projected real parameters p_X(A·y) (high-dim). Standardize the maximization-oriented targets.
        GpSearchSpace? space = _highDimensional ? new GpSearchSpace(embeddedSpace) : null;
        bool[] isCategorical = space?.IsCategorical ?? new bool[d];
        double[,] trainX = BuildTrainInputs(run, latentHistory, space, embeddedSpace);
        (double[] yStd, double incumbent) = Standardize(maxValues);

        // Algorithm 3: (re-)fit the length scale under the current upper bound on a periodic cadence.
        if (!state.Fitted || state.StepsSinceRefit >= RefitPeriod)
        {
            (state.LengthScale, state.CategoricalScale) = FitHyperparameters(
                trainX, yStd, isCategorical, state.UpperBound, HyperSeed(run, state.StepCount));
            state.Fitted = true;
            state.StepsSinceRefit = 0;
        }

        var gp = pendingLatents.Count == 0
            ? SquaredExponentialGp.Fit(trainX, yStd, isCategorical, state.LengthScale, state.CategoricalScale)
            : FitWithPendingLies(run, trainX, yStd, pendingLatents, space, embeddedSpace, isCategorical, state);

        // Maximize expected improvement over Y by minimizing −log-EI, with DIRECT + CMA-ES (§3.3).
        double h = _embedding.BoxHalfWidth;
        var lb = new double[d];
        var ub = new double[d];
        for (int j = 0; j < d; j++) { lb[j] = -h; ub[j] = h; }

        double NegAcq(double[] y)
        {
            double[] input = InputFor(run, y, space, embeddedSpace);
            (double mean, double var) = gp.Posterior(input);
            return -Acqf.LogEi(mean, var + AcqStabilizingNoise, incumbent);
        }

        (double[] yStar, double _) = GlobalMinimizer.Minimize(
            NegAcq, lb, ub, AcqSeed(run, state.StepCount), _acqDirectEvals, _acqCmaEvals);

        // Algorithm 3 exploit detection: a low posterior std at the chosen point means the step exploited.
        (double _, double vStar) = gp.Posterior(InputFor(run, yStar, space, embeddedSpace));
        if (Math.Sqrt(vStar) < ExploitThreshold) state.ExploitStreak++;
        else state.ExploitStreak = 0;

        state.StepsSinceRefit++;

        // After 5 consecutive exploiting steps, shrink the length-scale upper bound and re-fit for next time
        // (Algorithm 3, lines 10-14) — this curbs the over-smoothing that drives over-exploitation.
        if (state.ExploitStreak >= ExploitStreakToShrink)
        {
            state.UpperBound = Math.Max(ShrinkFactor * state.LengthScale, LengthScaleLower);
            state.ExploitStreak = 0;
            (state.LengthScale, state.CategoricalScale) = FitHyperparameters(
                trainX, yStd, isCategorical, state.UpperBound, HyperSeed(run, state.StepCount + 1));
            state.StepsSinceRefit = 0;
        }

        state.StepCount++;
        return yStar;
    }

    /// <summary>
    ///     Fits the acquisition's GP on the real observations plus the running trials' points, each told the worst
    ///     standardized value observed: the posterior variance at a running point collapses and its mean is low, so
    ///     expected improvement there is near zero and the next proposal moves away from it.
    /// </summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="trainX">The real observations' kernel-space inputs.</param>
    /// <param name="yStd">The real observations' standardized maximization values.</param>
    /// <param name="pendingLatents">The running trials' latent points.</param>
    /// <param name="space">The kernel-space search space (high-dimensional kernel), or null (low-dimensional).</param>
    /// <param name="embeddedSpace">The embedded real search space.</param>
    /// <param name="isCategorical">Per input column, whether the kernel treats it as categorical.</param>
    /// <param name="state">The run's fitted length scales.</param>
    /// <returns>The GP the acquisition is maximized over.</returns>
    private SquaredExponentialGp FitWithPendingLies(
        int run, double[,] trainX, double[] yStd, IReadOnlyList<double[]> pendingLatents, GpSearchSpace? space,
        OrderedDict<string, BaseDistribution> embeddedSpace, bool[] isCategorical, RunState state)
    {
        double[,] pendingX = BuildTrainInputs(run, pendingLatents, space, embeddedSpace);
        int n = trainX.GetLength(0), p = pendingX.GetLength(0), cols = trainX.GetLength(1);
        var x = new double[n + p, cols];
        var y = new double[n + p];
        double lie = yStd.Length == 0 ? 0.0 : yStd.Min();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < cols; j++)
                x[i, j] = trainX[i, j];
            y[i] = yStd[i];
        }

        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < cols; j++)
                x[n + i, j] = pendingX[i, j];
            y[n + i] = lie;
        }

        return SquaredExponentialGp.Fit(x, y, isCategorical, state.LengthScale, state.CategoricalScale);
    }

    /// <summary>Returns (creating on first use) the Algorithm-3 state for one embedding.</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <returns>The mutable run state.</returns>
    private RunState GetRunState(int run)
    {
        if (!_runs.TryGetValue(run, out RunState? state))
        {
            state = new RunState();
            _runs[run] = state;
        }
        return state;
    }

    /// <summary>
    ///     Fits the length scale ℓ (and, when categorical coordinates are present, the Hamming scale λ) by
    ///     maximizing the concentrated marginal likelihood over <c>[L, U] (× [L_λ, U_λ])</c> with DIRECT +
    ///     CMA-ES — the paper's §3.3 hyperparameter step. Maximization is done by minimizing the negative
    ///     log-likelihood.
    /// </summary>
    /// <param name="trainX">The GP training inputs in kernel space.</param>
    /// <param name="yStd">The standardized maximization-oriented targets.</param>
    /// <param name="isCategorical">Per-coordinate categorical flag.</param>
    /// <param name="upperBound">The current length-scale upper bound U.</param>
    /// <param name="seed">Seed for the CMA-ES half of the fit.</param>
    /// <returns>The fitted <c>(ℓ, λ)</c>.</returns>
    private (double LengthScale, double CategoricalScale) FitHyperparameters(
        double[,] trainX, double[] yStd, bool[] isCategorical, double upperBound, int seed)
    {
        bool hasCategorical = false;
        foreach (bool c in isCategorical) if (c) { hasCategorical = true; break; }

        double u = Math.Max(upperBound, LengthScaleLower * 1.0001);   // keep the box non-degenerate after shrinks
        if (hasCategorical)
        {
            (double[] theta, double _) = GlobalMinimizer.Minimize(
                p => -SquaredExponentialGp.ConcentratedLogLikelihood(trainX, yStd, isCategorical, p[0], p[1]),
                new[] { LengthScaleLower, CategoricalLower }, new[] { u, CategoricalUpper },
                seed, _hyperDirectEvals, _hyperCmaEvals);
            return (theta[0], theta[1]);
        }

        (double[] th, double _2) = GlobalMinimizer.Minimize(
            p => -SquaredExponentialGp.ConcentratedLogLikelihood(trainX, yStd, isCategorical, p[0], 0.0),
            new[] { LengthScaleLower }, new[] { u }, seed, _hyperDirectEvals, _hyperCmaEvals);
        return (th[0], 0.0);
    }

    /// <summary>Builds the GP training-input matrix in kernel space (latent <c>y</c> for the low-dimensional
    ///     kernel; normalized projected real parameters for the high-dimensional kernel).</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="latentHistory">The completed trials' latent points.</param>
    /// <param name="space">The normalized search space (high-dimensional kernel only; <c>null</c> otherwise).</param>
    /// <param name="embeddedSpace">The embedded real search space (high-dimensional kernel only).</param>
    /// <returns>The <c>n × dim</c> training-input matrix.</returns>
    private double[,] BuildTrainInputs(
        int run, IReadOnlyList<double[]> latentHistory, GpSearchSpace? space,
        OrderedDict<string, BaseDistribution> embeddedSpace)
    {
        int n = latentHistory.Count;
        if (space is null)
        {
            // Low-dimensional kernel: the inputs are the latent points themselves.
            int d = _embedding.Dim;
            var x = new double[n, d];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    x[i, j] = latentHistory[i][j];
            return x;
        }

        // High-dimensional kernel: project each latent point to the real parameters and normalize.
        var rows = new List<IReadOnlyDictionary<string, object?>>(n);
        foreach (double[] y in latentHistory)
            rows.Add(ProjectToExternal(run, y, embeddedSpace));
        return space.GetNormalizedParams(rows);
    }

    /// <summary>Maps a latent point to the GP's kernel-space input for a query (mirrors <see cref="BuildTrainInputs" /> per point).</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="y">The latent point.</param>
    /// <param name="space">The normalized search space (high-dimensional kernel only; <c>null</c> otherwise).</param>
    /// <param name="embeddedSpace">The embedded real search space (high-dimensional kernel only).</param>
    /// <returns>The kernel-space input vector (length <c>dim</c>).</returns>
    private double[] InputFor(
        int run, double[] y, GpSearchSpace? space, OrderedDict<string, BaseDistribution> embeddedSpace)
    {
        if (space is null)
            return y;   // low-dimensional kernel: identity
        double[,] norm = space.GetNormalizedParams(new[] { ProjectToExternal(run, y, embeddedSpace) });
        int dim = norm.GetLength(1);
        var row = new double[dim];
        for (int j = 0; j < dim; j++) row[j] = norm[0, j];
        return row;
    }

    /// <summary>Projects a latent point to external real-parameter values via <c>x = p_X(A·y)</c>, for the high-dimensional kernel.</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="y">The latent point.</param>
    /// <param name="embeddedSpace">The embedded real search space.</param>
    /// <returns>The external parameter values keyed by name.</returns>
    private IReadOnlyDictionary<string, object?> ProjectToExternal(
        int run, double[] y, OrderedDict<string, BaseDistribution> embeddedSpace)
    {
        var external = new Dictionary<string, object?>(embeddedSpace.Count);
        foreach (var (name, dist) in embeddedSpace)
            external[name] = _embedding.ToExternal(run, name, dist, y);
        return external;
    }

    /// <summary>Draws a uniform latent point in <c>Y = [−√d, √d]^d</c> from a deterministic, D-independent stream.</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="step">The current step count (part of the seed).</param>
    /// <returns>A uniform latent point in the box.</returns>
    private double[] RandomY(int run, int step)
    {
        var rng = np.random.RandomState();
        rng.seed(Mix(_baseSeed, (ulong)run * 0x9E3779B1u ^ (ulong)step));
        double h = _embedding.BoxHalfWidth;
        return rng.uniform(-h, h, new Shape(_embedding.Dim)).ToArray<double>();
    }

    /// <summary>Standardizes targets to zero mean and unit variance and returns the standardized incumbent (max).</summary>
    /// <param name="values">The maximization-oriented objective values.</param>
    /// <returns>The standardized values and the standardized incumbent (their maximum).</returns>
    private static (double[] Standardized, double Incumbent) Standardize(IReadOnlyList<double> values)
    {
        int n = values.Count;
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = values[i];
        double mean = MathUtil.Mean(arr);
        double std = Math.Max(1e-12, MathUtil.Std(arr));
        double incumbent = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            arr[i] = (arr[i] - mean) / std;
            if (arr[i] > incumbent) incumbent = arr[i];
        }
        return (arr, incumbent);
    }

    /// <summary>The acquisition optimizer's CMA-ES seed for one step — deterministic and D-independent.</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="step">The step count.</param>
    /// <returns>A non-negative 31-bit seed.</returns>
    private int AcqSeed(int run, int step) => (int)(Mix(_baseSeed, (ulong)run * 0x2545F491u ^ (ulong)step) & 0x7FFFFFFF);

    /// <summary>The hyperparameter fit's CMA-ES seed for one step — deterministic and D-independent.</summary>
    /// <param name="run">The embedding (run) index.</param>
    /// <param name="step">The step count.</param>
    /// <returns>A non-negative 31-bit seed.</returns>
    private int HyperSeed(int run, int step) => (int)(Mix(_baseSeed, (ulong)run * 0x85EBCA6Bu ^ (ulong)step ^ 0xD1B54A32u) & 0x7FFFFFFF);

    /// <summary>SplitMix64 finalizer folding the base seed and a counter into a well-scrambled 32-bit seed.</summary>
    /// <param name="seed">The base seed.</param>
    /// <param name="counter">The per-step counter.</param>
    /// <returns>A 32-bit seed.</returns>
    private static uint Mix(uint seed, ulong counter)
    {
        ulong z = ((ulong)seed << 32) ^ counter;
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (uint)(z & 0xFFFFFFFFUL);
    }
}

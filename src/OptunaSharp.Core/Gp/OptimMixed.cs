using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Gp;

/// <summary>
///     The mixed continuous/discrete acquisition-function optimizer — port of
///     <c>optuna/_gp/optim_mixed.py</c>. Given a fitted acqf (<see cref="IAcquisitionFunc" />), it
///     draws <c>n_preliminary_samples</c> scrambled-Sobol' points, picks the best plus a roulette
///     sample of promising ones as local-search start points, and runs an alternating local search —
///     L-BFGS-B gradient ascent over the continuous dims (with lengthscale preconditioning) and a
///     line/exhaustive search over each discrete dim — returning the best normalized point.
///
///     <para><b>Batched L-BFGS-B.</b> Optuna's <c>batched_lbfgsb</c> runs a batch of L-BFGS-B instances
///     that share batched function evaluations (a greenlet trick, or a sequential fallback). Both
///     paths give an identical per-row result — the batching only decides <em>when</em> evaluations
///     happen, not the L-BFGS-B trajectory — so this port runs the <b>sequential</b> per-row form over
///     the ported managed <see cref="LBfgsB" />. The GP path is tolerance-exact (ULP drift from scipy's
///     OpenBLAS BLAS + the erf/ucrt class in LogEI), the documented GP boundary.</para>
/// </summary>
internal static class OptimMixed
{
    private const int MaxIntExhaustiveSearchParams = 16;
    private const int Continuous = -1;

    /// <summary>
    ///     <c>optimize_acqf_mixed</c> — the entry point. Draws the preliminary QMC samples, chooses the
    ///     local-search start points (best + roulette), runs the mixed local searches and returns the
    ///     best <c>(normalized_params, acqf_value)</c>.
    /// </summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="warmstartNormalizedParamsArray">Previously promising normalized points appended to the initial search candidates.</param>
    /// <param name="nPreliminarySamples">The number of QMC candidates scored before selecting local-search starts.</param>
    /// <param name="nLocalSearch">The number of local searches initialized from the preliminary candidates.</param>
    /// <param name="tol">The relative stopping tolerance for the scalar minimizer.</param>
    /// <param name="rng">The random generator whose state is advanced when drawing samples.</param>
    /// <returns>The best normalized point found and its acquisition value.</returns>
    /// <exception cref="System.ArgumentException">No local-search start can be selected from sampled and supplied initial points.</exception>
    internal static (double[] Params, double Value) OptimizeAcqfMixed(
        IAcquisitionFunc acqf,
        double[,]? warmstartNormalizedParamsArray,
        int nPreliminarySamples,
        int nLocalSearch,
        double tol,
        NumPyRandom rng)
    {
        int dim = acqf.SearchSpace.Dim;
        double[,] warmstart = warmstartNormalizedParamsArray ?? new double[0, dim];
        int nWarmstart = warmstart.GetLength(0);
        if (nWarmstart > nLocalSearch - 1)
            throw new ArgumentException(
                "We must choose at least 1 best sampled point + given_initial_xs as start points.");

        double[,] sampledXs = acqf.SearchSpace.SampleNormalizedParams(nPreliminarySamples, rng);

        // Evaluate all values at initial samples.
        double[] fVals = acqf.EvalAcqfNoGrad(sampledXs);
        int maxI = Argmax(fVals);

        // A modified roulette wheel selection to pick the initial param for each local search.
        int nSamples = sampledXs.GetLength(0);
        var probs = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
            probs[i] = Math.Exp(fVals[i] - fVals[maxI]);
        probs[maxI] = 0.0;  // We already picked the best param, so remove it from roulette.
        double probsSum = MathUtil.PairwiseSum(probs);
        for (int i = 0; i < nSamples; i++)
            probs[i] /= probsSum;
        int nNonZeroProbs = 0;
        for (int i = 0; i < nSamples; i++)
            if (probs[i] > 0.0) nNonZeroProbs++;
        // n_additional_warmstart becomes smaller when study starts to converge.
        int nAdditionalWarmstart = Math.Min(nLocalSearch - nWarmstart - 1, nNonZeroProbs);
        if (nAdditionalWarmstart == nNonZeroProbs)
            Logger.Warning("Study already converged, so the number of local search is reduced.");

        var chosenIdxs = new List<int> { maxI };
        if (nAdditionalWarmstart > 0)
            chosenIdxs.AddRange(ChoiceNoReplaceWeighted(rng, nSamples, nAdditionalWarmstart, probs));

        // x_warmstarts = vstack([sampled_xs[chosen_idxs], warmstart]).
        int nStarts = chosenIdxs.Count + nWarmstart;
        var xWarmstarts = new double[nStarts, dim];
        for (int r = 0; r < chosenIdxs.Count; r++)
            for (int d = 0; d < dim; d++)
                xWarmstarts[r, d] = sampledXs[chosenIdxs[r], d];
        for (int r = 0; r < nWarmstart; r++)
            for (int d = 0; d < dim; d++)
                xWarmstarts[chosenIdxs.Count + r, d] = warmstart[r, d];

        (double[,] bestXs, double[] bestFvals) = LocalSearchMixedBatched(acqf, xWarmstarts, tol);
        int bestIdx = Argmax(bestFvals);
        var bestParam = new double[dim];
        for (int d = 0; d < dim; d++)
            bestParam[d] = bestXs[bestIdx, d];
        return (bestParam, bestFvals[bestIdx]);
    }

    /// <summary>
    ///     <c>local_search_mixed_batched</c> — alternate gradient ascent (continuous) and per-dim
    ///     discrete searches until no start point's best param changes (or <paramref name="maxIter" />
    ///     is hit). Rows converge independently; <c>remainingInds</c>/<c>lastChangedDims</c> track which
    ///     start points are still moving (the <c>CONTINUOUS</c> sentinel = the last change was a
    ///     gradient step), exactly as Optuna's masks do.
    /// </summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="xs0">The normalized starting points, one per local search.</param>
    /// <param name="tol">The relative stopping tolerance for the scalar minimizer.</param>
    /// <param name="maxIter">The maximum number of local-search iterations.</param>
    /// <returns>The best normalized point and acquisition value from each starting row.</returns>
    internal static (double[,] BestXs, double[] BestFvals) LocalSearchMixedBatched(
        IAcquisitionFunc acqf, double[,] xs0, double tol = 1e-4, int maxIter = 100)
    {
        GpSearchSpace space = acqf.SearchSpace;
        int[] contInds = space.ContinuousIndices;
        var lengthscales = new double[contInds.Length];
        for (int j = 0; j < contInds.Length; j++)
            lengthscales[j] = acqf.LengthScales[contInds[j]];
        int[] discreteIndices = space.DiscreteIndices;
        List<double[]> choicesOfDiscreteParams = space.GetChoicesOfDiscreteParams();
        var discreteXtols = new double[discreteIndices.Length];
        for (int k = 0; k < discreteIndices.Length; k++)
            discreteXtols[k] = MinDiff(choicesOfDiscreteParams[k]) / 4.0;

        int n = xs0.GetLength(0);
        int dim = xs0.GetLength(1);
        var bestXs = (double[,])xs0.Clone();
        double[] bestFvals = acqf.EvalAcqfNoGrad(bestXs);

        var lastChangedDims = new List<int>();
        var remainingInds = new List<int>();
        for (int i = 0; i < n; i++)
        {
            lastChangedDims.Add(Continuous);
            remainingInds.Add(i);
        }

        bool converged = false;
        for (int iter = 0; iter < maxIter && !converged; iter++)
        {
            // ---- gradient ascent over the continuous dims for the remaining rows -----------------
            (double[,] gaXs, double[] gaF, bool[] gaUpdated) = GradientAscentBatched(
                acqf, SubRows(bestXs, remainingInds), Sub(bestFvals, remainingInds),
                contInds, lengthscales, tol);
            WriteRows(bestXs, remainingInds, gaXs);
            WriteSub(bestFvals, remainingInds, gaF);
            for (int k = 0; k < remainingInds.Count; k++)
                if (gaUpdated[k]) lastChangedDims[k] = Continuous;

            // ---- per-discrete-dim line/exhaustive search -----------------------------------------
            for (int di = 0; di < discreteIndices.Length; di++)
            {
                int i = discreteIndices[di];
                double[] choices = choicesOfDiscreteParams[di];
                double xtol = discreteXtols[di];

                // Drop rows whose last change was this dim i (converged on it).
                DropWhere(remainingInds, lastChangedDims, d => d == i);
                if (remainingInds.Count == 0)
                    return (bestXs, bestFvals);

                (double[,] dXs, double[] dF, bool[] dUpdated) = LocalSearchDiscreteBatched(
                    acqf, SubRows(bestXs, remainingInds), Sub(bestFvals, remainingInds), i, choices, xtol);
                WriteRows(bestXs, remainingInds, dXs);
                WriteSub(bestFvals, remainingInds, dF);
                for (int k = 0; k < remainingInds.Count; k++)
                    if (dUpdated[k]) lastChangedDims[k] = i;
            }

            // Parameters not changed from the beginning or last changed dimension is continuous.
            DropWhere(remainingInds, lastChangedDims, d => d == Continuous);
            if (remainingInds.Count == 0)
                converged = true;
        }

        if (!converged)
            Logger.Warning("local_search_mixed: Local search did not converge.");
        return (bestXs, bestFvals);
    }

    /// <summary>
    ///     <c>_gradient_ascent_batched</c> — L-BFGS-B gradient ascent over the continuous dims with the
    ///     lengthscale preconditioning (optimize w.r.t. <c>z = x / l</c> so the isotropic-kernel Hessian
    ///     is well-conditioned; the domain becomes <c>[0, 1/l]</c>). Runs one managed
    ///     <see cref="LBfgsB" /> per row — the sequential form of <c>batched_lbfgsb</c>.
    /// </summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="initialParamsBatched">The initial normalized coordinates, one local-search start per row.</param>
    /// <param name="initialFvals">The acquisition value at each corresponding starting row.</param>
    /// <param name="continuousIndices">The coordinate indices that permit continuous gradient steps.</param>
    /// <param name="lengthscales">Kernel length scales used to precondition the gradient step.</param>
    /// <param name="tol">The relative stopping tolerance for the scalar minimizer.</param>
    /// <returns>The updated points, their acquisition values, and flags identifying improved starts.</returns>
    internal static (double[,] Xs, double[] Fvals, bool[] Updated) GradientAscentBatched(
        IAcquisitionFunc acqf, double[,] initialParamsBatched, double[] initialFvals,
        int[] continuousIndices, double[] lengthscales, double tol)
    {
        int b = initialParamsBatched.GetLength(0);
        int dim = initialParamsBatched.GetLength(1);
        int nc = continuousIndices.Length;
        if (nc == 0)
            return (initialParamsBatched, initialFvals, new bool[b]);

        var bounds = new (double Lower, double Upper)[nc];
        for (int j = 0; j < nc; j++)
            bounds[j] = (0.0, 1.0 / lengthscales[j]);

        double pgtol = Math.Sqrt(tol);
        var xsOpt = (double[,])initialParamsBatched.Clone();
        var fvalsOpt = new double[b];
        var nIterations = new int[b];

        for (int i = 0; i < b; i++)
        {
            var fixedParams = new double[dim];
            for (int d = 0; d < dim; d++)
                fixedParams[d] = initialParamsBatched[i, d];
            var x0 = new double[nc];
            for (int j = 0; j < nc; j++)
                x0[j] = initialParamsBatched[i, continuousIndices[j]] / lengthscales[j];

            // negative_acqf_with_grad for this row: scale z back to x = z*l, then flip signs (scipy
            // minimizes) and chain the outer scale: dg/dz = df/dx * l.
            (double, double[]) FuncAndGrad(double[] scaledX)
            {
                var nextParams = (double[])fixedParams.Clone();
                for (int j = 0; j < nc; j++)
                    nextParams[continuousIndices[j]] = scaledX[j] * lengthscales[j];
                (double val, double[] grad) = acqf.EvalAcqfWithGrad(nextParams);
                var negGrad = new double[nc];
                for (int j = 0; j < nc; j++)
                    negGrad[j] = -grad[continuousIndices[j]] * lengthscales[j];
                return (-val, negGrad);
            }

            LBfgsB.Result res = LBfgsB.Minimize(
                FuncAndGrad, x0, bounds, m: 10, factr: 1e7, pgtol: pgtol,
                maxfun: 15000, maxiter: 200, maxls: 20);
            for (int j = 0; j < nc; j++)
                xsOpt[i, continuousIndices[j]] = res.X[j] * lengthscales[j];
            fvalsOpt[i] = -res.F;
            nIterations[i] = res.NumIterations;
        }

        // is_updated = (fvals_opt > initial_fvals) & (n_iterations > 0); return updated rows else initial.
        var updated = new bool[b];
        var outXs = (double[,])initialParamsBatched.Clone();
        var outF = new double[b];
        for (int i = 0; i < b; i++)
        {
            updated[i] = fvalsOpt[i] > initialFvals[i] && nIterations[i] > 0;
            if (updated[i])
            {
                for (int d = 0; d < dim; d++)
                    outXs[i, d] = xsOpt[i, d];
                outF[i] = fvalsOpt[i];
            }
            else
            {
                outF[i] = initialFvals[i];
            }
        }
        return (outXs, outF, updated);
    }

    // ---- discrete searches -----------------------------------------------------------------------

    /// <summary><c>_local_search_discrete_batched</c> — loop the single-row discrete search over the batch.</summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="initialParamsBatched">The initial normalized coordinates, one local-search start per row.</param>
    /// <param name="initialFvals">The acquisition value at each corresponding starting row.</param>
    /// <param name="paramIdx">The zero-based parameter index in the model's parameter order.</param>
    /// <param name="choices">The admissible values of the discrete or categorical coordinate.</param>
    /// <param name="xtol">The coordinate tolerance used by the discrete line search.</param>
    /// <returns>The updated discrete-coordinate points, acquisition values, and improvement flags.</returns>
    private static (double[,] Xs, double[] Fvals, bool[] Updated) LocalSearchDiscreteBatched(
        IAcquisitionFunc acqf, double[,] initialParamsBatched, double[] initialFvals,
        int paramIdx, double[] choices, double xtol)
    {
        int b = initialParamsBatched.GetLength(0);
        int dim = initialParamsBatched.GetLength(1);
        var bestXs = (double[,])initialParamsBatched.Clone();
        var bestFvals = (double[])initialFvals.Clone();
        var updated = new bool[b];
        for (int i = 0; i < b; i++)
        {
            var row = new double[dim];
            for (int d = 0; d < dim; d++)
                row[d] = initialParamsBatched[i, d];
            (double[] bp, double bf, bool up) = LocalSearchDiscrete(
                acqf, row, bestFvals[i], paramIdx, choices, xtol);
            for (int d = 0; d < dim; d++)
                bestXs[i, d] = bp[d];
            bestFvals[i] = bf;
            updated[i] = up;
        }
        return (bestXs, bestFvals, updated);
    }

    /// <summary><c>_local_search_discrete</c> — exhaustive for categoricals / small grids, else line search.</summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="initialParams">The normalized coordinates from which the local search starts.</param>
    /// <param name="initialFval">The acquisition value already evaluated at the initial coordinates.</param>
    /// <param name="paramIdx">The zero-based parameter index in the model's parameter order.</param>
    /// <param name="choices">The admissible values of the discrete or categorical coordinate.</param>
    /// <param name="xtol">The coordinate tolerance used by the discrete line search.</param>
    /// <returns>The best point found by discrete-coordinate search, its value, and whether it improved the start.</returns>
    private static (double[] Params, double Fval, bool Updated) LocalSearchDiscrete(
        IAcquisitionFunc acqf, double[] initialParams, double initialFval,
        int paramIdx, double[] choices, double xtol)
    {
        bool isCategorical = acqf.SearchSpace.IsCategorical[paramIdx];
        if (isCategorical || choices.Length <= MaxIntExhaustiveSearchParams)
            return ExhaustiveSearch(acqf, initialParams, initialFval, paramIdx, choices);
        return DiscreteLineSearch(acqf, initialParams, initialFval, paramIdx, choices, xtol);
    }

    /// <summary><c>_exhaustive_search</c> — evaluate every off-current choice; take the best if it improves.</summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="initialParams">The normalized coordinates from which the local search starts.</param>
    /// <param name="initialFval">The acquisition value already evaluated at the initial coordinates.</param>
    /// <param name="paramIdx">The zero-based parameter index in the model's parameter order.</param>
    /// <param name="choices">The admissible values of the discrete or categorical coordinate.</param>
    /// <returns>The best admissible point along the selected coordinate, its value, and whether it improved the start.</returns>
    private static (double[] Params, double Fval, bool Updated) ExhaustiveSearch(
        IAcquisitionFunc acqf, double[] initialParams, double initialFval, int paramIdx, double[] choices)
    {
        if (choices.Length == 1)
            return (initialParams, initialFval, false);

        var except = new List<double>();
        double current = initialParams[paramIdx];
        foreach (double c in choices)
            if (c != current)
                except.Add(c);

        int dim = initialParams.Length;
        var allParams = new double[except.Count, dim];
        for (int r = 0; r < except.Count; r++)
        {
            for (int d = 0; d < dim; d++)
                allParams[r, d] = initialParams[d];
            allParams[r, paramIdx] = except[r];
        }
        double[] fvals = acqf.EvalAcqfNoGrad(allParams);
        int bestIdx = Argmax(fvals);
        if (fvals[bestIdx] > initialFval)
        {
            var best = new double[dim];
            for (int d = 0; d < dim; d++)
                best[d] = allParams[bestIdx, d];
            return (best, fvals[bestIdx], true);
        }
        return (initialParams, initialFval, false);
    }

    /// <summary>
    ///     <c>_discrete_line_search</c> — Brent line search over the piecewise-linear interpolation of
    ///     the acqf between grid points (EI is often high between a grid and its neighbour), snapping
    ///     the result to the nearest grid index.
    /// </summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="initialParams">The normalized coordinates from which the local search starts.</param>
    /// <param name="initialFval">The acquisition value already evaluated at the initial coordinates.</param>
    /// <param name="paramIdx">The zero-based parameter index in the model's parameter order.</param>
    /// <param name="grids">The admissible normalized values for each discrete coordinate.</param>
    /// <param name="xtol">The coordinate tolerance used by the discrete line search.</param>
    /// <returns>The best grid point found along the coordinate, its value, and whether it improved the start.</returns>
    private static (double[] Params, double Fval, bool Updated) DiscreteLineSearch(
        IAcquisitionFunc acqf, double[] initialParams, double initialFval,
        int paramIdx, double[] grids, double xtol)
    {
        if (grids.Length == 1)
            return (initialParams, initialFval, false);

        int FindNearestIndex(double x)
        {
            int i = Math.Clamp(BisectLeft(grids, x), 1, grids.Length - 1);
            return Math.Abs(x - grids[i - 1]) < Math.Abs(x - grids[i]) ? i - 1 : i;
        }

        int currentChoiceI = FindNearestIndex(initialParams[paramIdx]);
        var negFvalCache = new Dictionary<int, double> { [currentChoiceI] = -initialFval };
        var normalizedParams = (double[])initialParams.Clone();

        double NegativeAcqfWithCache(int i)
        {
            if (negFvalCache.TryGetValue(i, out double cached))
                return cached;
            normalizedParams[paramIdx] = grids[i];
            double negval = -acqf.EvalAcqfNoGrad(normalizedParams);
            negFvalCache[i] = negval;
            return negval;
        }

        double InterpolatedNegativeAcqf(double x)
        {
            if (x < grids[0] || x > grids[^1])
                return double.PositiveInfinity;
            int right = Math.Clamp(BisectLeft(grids, x), 1, grids.Length - 1);
            int left = right - 1;
            double negLeft = NegativeAcqfWithCache(left);
            double negRight = NegativeAcqfWithCache(right);
            double wLeft = (grids[right] - x) / (grids[right] - grids[left]);
            double wRight = 1.0 - wLeft;
            return wLeft * negLeft + wRight * negRight;
        }

        const double eps = 1e-12;
        // Bracket (grids[0]-EPS, grids[current], grids[-1]+EPS): f = (inf, -fval, inf), a valid bracket.
        double resX = BrentMinimizer.MinimizeScalar(
            InterpolatedNegativeAcqf, grids[0] - eps, grids[currentChoiceI], grids[^1] + eps, xtol);
        int optIdx = FindNearestIndex(resX);
        double fvalOpt = -NegativeAcqfWithCache(optIdx);

        // Both conditions are checked because of numerical errors.
        if (optIdx != currentChoiceI && fvalOpt > initialFval)
        {
            normalizedParams[paramIdx] = grids[optIdx];
            return (normalizedParams, fvalOpt, true);
        }
        return (initialParams, initialFval, false);
    }

    // ---- numpy helpers ---------------------------------------------------------------------------

    /// <summary>
    ///     NumPy <c>RandomState.choice(n, size, replace=False, p=p)</c>: the weighted
    ///     sampling-without-replacement loop — draw <c>random_sample(size - n_uniq)</c> uniforms, invert
    ///     the (re-normalized, zeroed-for-found) cdf via <c>searchsorted(side='right')</c>, dedup
    ///     preserving first appearance, until <paramref name="size" /> unique indices are found. Draw
    ///     count and RNG state advance match numpy exactly (verified), keeping the sampler in lock-step.
    /// </summary>
    /// <param name="rng">The random generator whose state is advanced when drawing samples.</param>
    /// <param name="n">The number of candidate indices in the population.</param>
    /// <param name="size">The number of distinct indices to draw.</param>
    /// <param name="p">The probability weight of each candidate index; at least size entries must be positive.</param>
    /// <returns>Distinct sampled indices in draw order.</returns>
    /// <exception cref="System.ArgumentException">Fewer candidate weights are positive than the requested number of distinct draws.</exception>
    internal static int[] ChoiceNoReplaceWeighted(NumPyRandom rng, int n, int size, double[] p)
    {
        int nonzero = 0;
        for (int i = 0; i < n; i++)
            if (p[i] > 0.0) nonzero++;
        if (nonzero < size)
            throw new ArgumentException("Fewer non-zero entries in p than size");

        var pc = (double[])p.Clone();
        int nUniq = 0;
        var found = new int[size];
        var seen = new HashSet<int>();
        while (nUniq < size)
        {
            int k = size - nUniq;
            var x = rng.random_sample((long)k).ToArray<double>();
            if (nUniq > 0)
                for (int j = 0; j < nUniq; j++)
                    pc[found[j]] = 0.0;
            var cdf = new double[n];
            double acc = 0.0;
            for (int i = 0; i < n; i++)
            {
                acc += pc[i];  // np.cumsum: sequential prefix sum.
                cdf[i] = acc;
            }
            double last = cdf[n - 1];
            for (int i = 0; i < n; i++)
                cdf[i] /= last;

            // new = searchsorted(cdf, x, side='right'); then dedup preserving first-appearance order
            // (== np.unique(return_index) + sort(indices) + take).
            for (int j = 0; j < k; j++)
            {
                int idx = BisectRight(cdf, x[j]);
                if (seen.Add(idx))
                {
                    found[nUniq] = idx;
                    nUniq++;
                }
            }
        }
        return found;
    }

    /// <summary>
    ///     NumPy <c>np.argmax</c> through NumSharp — the optimizer's <c>np.argmax(f_vals)</c>, <c>np.argmax(best_fvals)</c>
    ///     and <c>np.argmax(fvals)</c>: the first index of the maximum, or of the first NaN when one is NaN.
    /// </summary>
    /// <param name="a">The acquisition values of the evaluated points; at least one.</param>
    /// <returns>The index of the first NaN if any value is NaN, else the first index holding the largest value.</returns>
    /// <remarks>
    ///     NumPy treats a NaN as maximal; NaN acquisition values come from a NaN threshold (QA-53) or overflowing
    ///     posteriors. The former strict <c>&gt;</c> loop skipped NaNs where they were not first. Each call follows an
    ///     acquisition evaluation over all the points, which costs far more than the NumSharp call.
    /// </remarks>
    /// <exception cref="NumSharp.ValueError"><paramref name="a" /> is empty (NumPy's <c>attempt to get argmax of an empty sequence</c>).</exception>
    private static int Argmax(double[] a)
    {
        using NDArray values = np.array(a, copy: false);
        using NDArray index = np.argmax(values);
        return (int)(long)index;
    }

    /// <summary><c>np.min(np.diff(choices), initial=np.inf)</c> — smallest consecutive gap, +∞ if &lt; 2 points.</summary>
    /// <param name="choices">The admissible values of the discrete or categorical coordinate.</param>
    /// <returns>The smallest adjacent spacing in the sorted coordinate grid.</returns>
    private static double MinDiff(double[] choices)
    {
        double min = double.PositiveInfinity;
        for (int i = 1; i < choices.Length; i++)
            min = Math.Min(min, choices[i] - choices[i - 1]);
        return min;
    }

    /// <summary><c>np.searchsorted(a, v, side='left')</c> — first index <c>i</c> with <c>a[i] &gt;= v</c>.</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="v">The values used by the numerical reduction or transformation.</param>
    /// <returns>The first insertion index that preserves sorted order before equal elements.</returns>
    private static int BisectLeft(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (a[mid] < v) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /// <summary><c>np.searchsorted(a, v, side='right')</c> — first index <c>i</c> with <c>a[i] &gt; v</c>.</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="v">The values used by the numerical reduction or transformation.</param>
    /// <returns>The first insertion index after any equal elements.</returns>
    private static int BisectRight(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (a[mid] <= v) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    // ---- sub-array bookkeeping (the fancy-index masks in local_search_mixed_batched) -------------

    /// <summary>Returns a new matrix containing the selected rows in index order.</summary>
    /// <param name="full">The complete vector or matrix from which a subset is read or updated.</param>
    /// <param name="inds">The zero-based indices of the values or rows to select.</param>
    /// <returns>A new matrix containing the selected rows in index order.</returns>
    private static double[,] SubRows(double[,] full, List<int> inds)
    {
        int dim = full.GetLength(1);
        var sub = new double[inds.Count, dim];
        for (int k = 0; k < inds.Count; k++)
            for (int d = 0; d < dim; d++)
                sub[k, d] = full[inds[k], d];
        return sub;
    }

    /// <summary>Returns a new vector containing the selected coordinates in index order.</summary>
    /// <param name="full">The complete vector or matrix from which a subset is read or updated.</param>
    /// <param name="inds">The zero-based indices of the values or rows to select.</param>
    /// <returns>A new vector containing the selected coordinates in index order.</returns>
    private static double[] Sub(double[] full, List<int> inds)
    {
        var sub = new double[inds.Count];
        for (int k = 0; k < inds.Count; k++)
            sub[k] = full[inds[k]];
        return sub;
    }

    /// <summary>Copies updated local-search rows back into the full candidate matrix.</summary>
    /// <param name="full">The complete vector or matrix from which a subset is read or updated.</param>
    /// <param name="inds">The zero-based indices of the values or rows to select.</param>
    /// <param name="sub">The replacement values for the selected coordinate subset.</param>
    private static void WriteRows(double[,] full, List<int> inds, double[,] sub)
    {
        int dim = full.GetLength(1);
        for (int k = 0; k < inds.Count; k++)
            for (int d = 0; d < dim; d++)
                full[inds[k], d] = sub[k, d];
    }

    /// <summary>Copies updated coordinates into their selected positions in the full parameter vector.</summary>
    /// <param name="full">The complete vector or matrix from which a subset is read or updated.</param>
    /// <param name="inds">The zero-based indices of the values or rows to select.</param>
    /// <param name="sub">The replacement values for the selected coordinate subset.</param>
    private static void WriteSub(double[] full, List<int> inds, double[] sub)
    {
        for (int k = 0; k < inds.Count; k++)
            full[inds[k]] = sub[k];
    }

    /// <summary>Drop the positions of <paramref name="inds" />/<paramref name="dims" /> where <paramref name="pred" /> holds (the <c>[~is_converged]</c> masks; both lists stay aligned).</summary>
    /// <param name="inds">The zero-based indices of the values or rows to select.</param>
    /// <param name="dims">The coordinate indices to update.</param>
    /// <param name="pred">The predicate used to locate a boundary in the ordered values.</param>
    private static void DropWhere(List<int> inds, List<int> dims, Func<int, bool> pred)
    {
        int w = 0;
        for (int r = 0; r < inds.Count; r++)
        {
            if (pred(dims[r]))
                continue;
            inds[w] = inds[r];
            dims[w] = dims[r];
            w++;
        }
        inds.RemoveRange(w, inds.Count - w);
        dims.RemoveRange(w, dims.Count - w);
    }

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna._gp.optim_mixed");
}

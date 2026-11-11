// TuRBO — Trust-Region Bayesian Optimization (Eriksson et al., NeurIPS 2019), the TR mechanics core.
//
// Reference:
//   David Eriksson, Michael Pearce, Jacob Gardner, Ryan D. Turner, Matthias Poloczek.
//     "Scalable Global Optimization via Local Bayesian Optimization." NeurIPS 2019 (arXiv:1910.01739), §2.
//   Defaults pinned against the BoTorch TuRBO tutorial (the canonical reference implementation), except
//   the success tolerance, which follows the paper supplement (τ_succ = 3).
//
// TuRBO abandons a single global GP surrogate — brittle in high dimension — for a LOCAL GP inside a
// trust region (TR): a hyperrectangle centered on the incumbent whose per-dimension side lengths track
// the fitted GP length scales (constant total volume L^d). Candidates are drawn by Thompson sampling
// over a Sobol perturbation set restricted to the TR; the TR expands after consecutive successes, shrinks
// after consecutive failures, and restarts once it collapses below a floor. This class holds the pure TR
// geometry + Thompson-sampling math (no Optuna/GP wiring); `TurboSampler` drives it. TuRBO has no `cmaes`/
// Optuna/OptunaHub counterpart, so — like `LmCma` / `De` — this is a native implementation validated by
// convergence, not bit-matched to an upstream trajectory. RNG is NumSharp's MT19937 (never System.Random).

using System;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Turbo;

/// <summary>
///     The reconstructed trust-region state after folding the observed objective sequence — TuRBO's
///     analogue of a persisted optimizer, recomputed each trial from the (deterministic) history rather
///     than serialized. It identifies the current local run (restart), its base side length, and the
///     success/failure counters that drive the next resize.
/// </summary>
internal readonly struct TurboTrState
{
    /// <summary>The current base side length L of the trust region (per-dimension sides rescale from it).</summary>
    public double Length { get; }

    /// <summary>Index into the ordered completed-trial array where the current restart's first trial sits;
    ///     the local GP fits on trials from here onward.</summary>
    public int RestartStart { get; }

    /// <summary>Consecutive-success count toward the next expansion (reset on any resize).</summary>
    public int SuccessCounter { get; }

    /// <summary>Consecutive-failure count toward the next shrink (reset on any resize).</summary>
    public int FailureCounter { get; }

    /// <summary>Number of restarts triggered so far (a diagnostics/testing signal; the TR collapsed this many times).</summary>
    public int RestartCount { get; }

    /// <summary>Captures one reconstructed trust-region state.</summary>
    /// <param name="length">The current base side length L.</param>
    /// <param name="restartStart">The ordered-history index where the current restart begins.</param>
    /// <param name="successCounter">The consecutive-success count.</param>
    /// <param name="failureCounter">The consecutive-failure count.</param>
    /// <param name="restartCount">The number of restarts triggered so far.</param>
    public TurboTrState(double length, int restartStart, int successCounter, int failureCounter, int restartCount)
    {
        Length = length;
        RestartStart = restartStart;
        SuccessCounter = successCounter;
        FailureCounter = failureCounter;
        RestartCount = restartCount;
    }
}

/// <summary>
///     The trust-region mechanics of TuRBO-1 (single trust region): the deterministic state fold, the
///     length-scale-weighted TR box, the Sobol perturbation candidate set, and Thompson sampling over
///     candidates. Everything operates in the transformed <c>[0, 1]^d</c> search box; the caller
///     (<see cref="TurboSampler" />) fits the local GP and maps candidates to external parameters.
///
///     <para><b>Orientation.</b> All scores here are <b>maximization-oriented</b> (higher is better) — the
///     caller negates a minimizing objective before folding, so the incumbent is the largest score and a
///     "success" is a score improvement.</para>
/// </summary>
internal sealed class TurboOptimizer
{
    private readonly double _lengthInit;
    private readonly double _lengthMin;
    private readonly double _lengthMax;
    private readonly int _tauSucc;
    private readonly int _tauFail;
    private readonly double _successTol;

    /// <param name="lengthInit">Initial base side length L on a fresh restart.</param>
    /// <param name="lengthMin">Lower floor; when L falls below this the trust region is discarded and restarted.</param>
    /// <param name="lengthMax">Upper cap on the base side length under expansion.</param>
    /// <param name="tauSucc">Consecutive successes required to double L.</param>
    /// <param name="tauFail">Consecutive failures required to halve L.</param>
    /// <param name="successTol">Relative improvement over the incumbent required to count as a success
    ///     (<c>score &gt; best + successTol·|best|</c>).</param>
    /// <summary>Configures the trust-region resize schedule and success criterion.</summary>
    /// <exception cref="System.ArgumentException">A length bound is non-positive/ill-ordered or a tolerance is negative.</exception>
    internal TurboOptimizer(
        double lengthInit, double lengthMin, double lengthMax, int tauSucc, int tauFail, double successTol)
    {
        if (!(lengthMin > 0 && lengthInit >= lengthMin && lengthMax >= lengthInit))
            throw new ArgumentException("Require 0 < lengthMin ≤ lengthInit ≤ lengthMax.");
        if (tauSucc < 1 || tauFail < 1)
            throw new ArgumentException("Success/failure tolerances must be at least 1.");
        if (successTol < 0)
            throw new ArgumentException("successTol must be non-negative.");
        _lengthInit = lengthInit;
        _lengthMin = lengthMin;
        _lengthMax = lengthMax;
        _tauSucc = tauSucc;
        _tauFail = tauFail;
        _successTol = successTol;
    }

    /// <summary>
    ///     Reconstructs the current trust-region state by folding the maximization-oriented objective
    ///     sequence in evaluation order. The first <paramref name="nInit" /> trials of each restart seed the
    ///     incumbent (no success/failure counting); subsequent trials count as a success (score improves the
    ///     incumbent) or failure and drive the resize schedule; when L falls below the floor the fold
    ///     discards the region and the following trials begin a fresh restart. This is a pure left-fold, so
    ///     it is prefix-consistent with the decisions made during the run and needs no persisted state.
    /// </summary>
    /// <param name="scores">Maximization-oriented objective scores in evaluation order; a non-finite entry
    ///     (a failed/blown-up trial) is treated as the worst possible score and never improves the incumbent.</param>
    /// <param name="nInit">Per-restart number of initialization trials before the local GP takes over.</param>
    /// <returns>The current base length, the current restart's start index, the resize counters, and the restart count.</returns>
    internal TurboTrState ReconstructState(double[] scores, int nInit)
    {
        double length = _lengthInit;
        double best = double.NegativeInfinity;
        int restartStart = 0;      // ordered index where the current restart begins
        int countInRestart = 0;    // trials seen in the current restart
        int succ = 0, fail = 0, restarts = 0;

        for (int i = 0; i < scores.Length; i++)
        {
            double y = scores[i];
            if (countInRestart < nInit)
            {
                // Initialization trial: seed the incumbent, do not count success/failure.
                if (y > best) best = y;
                countInRestart++;
                continue;
            }

            // GP-proposed trial: success iff it improves the incumbent by the relative tolerance. When the
            // incumbent is still −∞ (all init trials were non-finite) any finite score is an improvement.
            bool improved = double.IsNegativeInfinity(best)
                ? IsFinite(y)
                : y > best + _successTol * Math.Abs(best);
            if (improved) { succ++; fail = 0; }
            else { succ = 0; fail++; }
            if (y > best) best = y;
            countInRestart++;

            if (succ == _tauSucc) { length = Math.Min(length * 2.0, _lengthMax); succ = 0; }
            else if (fail == _tauFail) { length /= 2.0; fail = 0; }

            if (length < _lengthMin)
            {
                // Discard and restart: the next trial begins a new local run with a fresh incumbent.
                length = _lengthInit;
                best = double.NegativeInfinity;
                succ = 0; fail = 0; countInRestart = 0;
                restartStart = i + 1;
                restarts++;
            }
        }

        return new TurboTrState(length, restartStart, succ, fail, restarts);
    }

    /// <summary>
    ///     Selects the trust-region center x⋆ among the current local run's points (paper §2): the best
    ///     <b>observation</b> when the objective is noise-free (<paramref name="deterministic" />), otherwise
    ///     the point with the best <b>posterior mean</b> under the fitted GP. The noisy rule exists so that a
    ///     single lucky (or unlucky) noise draw cannot anchor the whole trust region on a spurious point —
    ///     the observed best may be an outlier the GP smooths away. "Best" is the largest value here: scores
    ///     are maximization-oriented, and because the posterior mean is fitted on the affinely standardized
    ///     score its argmax is unchanged, so this maximizing argmax realizes the paper's "smallest posterior
    ///     mean" rule stated in its minimization convention.
    /// </summary>
    /// <param name="observedScores">The maximization-oriented observed score of each point in the current run, length m ≥ 1.</param>
    /// <param name="posteriorMean">The GP posterior mean at those same points; ignored (may be <c>null</c>) when
    ///     <paramref name="deterministic" /> is true, otherwise required with the same length as <paramref name="observedScores" />.</param>
    /// <param name="deterministic">True to center on the best observation (noise-free case); false to center on the best posterior mean (noisy case).</param>
    /// <returns>The index of the chosen center point (the first argmax on a tie).</returns>
    /// <exception cref="System.ArgumentException"><paramref name="observedScores" /> is empty, or the noisy branch was
    ///     requested without a <paramref name="posteriorMean" /> of matching length.</exception>
    internal static int SelectCenterIndex(double[] observedScores, double[]? posteriorMean, bool deterministic)
    {
        int m = observedScores.Length;
        if (m < 1)
            throw new ArgumentException("observedScores must be non-empty.", nameof(observedScores));

        // Pick the key to argmax over: observations (noise-free) or the GP posterior mean (noisy).
        double[] key;
        if (deterministic)
        {
            key = observedScores;
        }
        else
        {
            if (posteriorMean is null || posteriorMean.Length != m)
                throw new ArgumentException(
                    "posteriorMean must be supplied with the same length as observedScores in the noisy case.",
                    nameof(posteriorMean));
            key = posteriorMean;
        }

        int best = 0;
        for (int i = 1; i < m; i++)
            if (key[i] > key[best]) best = i;
        return best;
    }

    /// <summary>
    ///     The trust-region hyperrectangle around <paramref name="xCenter" /> in normalized <c>[0, 1]^d</c>:
    ///     each side is rescaled by the GP length scale, <c>side_i = λ_i · L / (Π_j λ_j)^{1/d}</c> (so the
    ///     total volume stays <c>L^d</c>), and clipped to the unit box. Longer length scales (flatter
    ///     directions) get wider sides.
    /// </summary>
    /// <param name="xCenter">The incumbent in normalized coordinates (the TR center), length d.</param>
    /// <param name="lengthScales">The fitted GP per-dimension length scales (all positive), length d.</param>
    /// <param name="length">The current base side length L.</param>
    /// <returns>The per-dimension lower and upper bounds of the trust region, each clipped to <c>[0, 1]</c>.</returns>
    /// <exception cref="System.ArgumentException">The center and length-scale lengths differ.</exception>
    internal (double[] Lb, double[] Ub) TrustRegionBounds(double[] xCenter, double[] lengthScales, double length)
    {
        int d = xCenter.Length;
        if (lengthScales.Length != d)
            throw new ArgumentException("lengthScales must match the center dimension.");

        // Geometric mean of the length scales: exp(mean(log λ)). Used to normalize the per-dim weights so
        // their product is 1 and the box volume is exactly L^d regardless of the raw length-scale magnitudes.
        double logSum = 0.0;
        for (int j = 0; j < d; j++) logSum += Math.Log(lengthScales[j]);
        double geoMean = Math.Exp(logSum / d);

        var lb = new double[d];
        var ub = new double[d];
        for (int j = 0; j < d; j++)
        {
            double weight = lengthScales[j] / geoMean;   // Π weight = 1
            double half = 0.5 * weight * length;
            lb[j] = Math.Max(0.0, xCenter[j] - half);
            ub[j] = Math.Min(1.0, xCenter[j] + half);
        }
        return (lb, ub);
    }

    /// <summary>
    ///     The Thompson-sampling candidate set inside the trust region (BoTorch TuRBO tutorial's
    ///     <c>generate_batch</c> for <c>acqf="ts"</c>): draw a scrambled Sobol' set in the TR box, then, for
    ///     each candidate, perturb only a random subset of coordinates (each chosen with probability
    ///     <c>min(20/d, 1)</c>, at least one) and hold the rest at the incumbent. In low dimension
    ///     (<c>d ≤ 20</c>) every coordinate is perturbed, so candidates fill the TR box; in high dimension
    ///     the sparse mask keeps candidates near the incumbent along most axes, which is what makes TuRBO
    ///     scale.
    /// </summary>
    /// <param name="xCenter">The incumbent in normalized coordinates, length d.</param>
    /// <param name="lb">Per-dimension lower bounds of the trust region (from <see cref="TrustRegionBounds" />).</param>
    /// <param name="ub">Per-dimension upper bounds of the trust region.</param>
    /// <param name="nCandidates">Number of candidate rows to generate.</param>
    /// <param name="rng">The per-trial MT19937 stream (seeded deterministically by the caller).</param>
    /// <returns>The <c>nCandidates × d</c> matrix of normalized candidate points inside the trust region.</returns>
    /// <exception cref="System.ArgumentException">The bound lengths differ from the center dimension, or the candidate count is non-positive.</exception>
    internal double[,] GenerateCandidates(
        double[] xCenter, double[] lb, double[] ub, int nCandidates, NumPyRandom rng)
    {
        int d = xCenter.Length;
        if (lb.Length != d || ub.Length != d)
            throw new ArgumentException("Bound lengths must match the center dimension.");
        if (nCandidates < 1)
            throw new ArgumentException("nCandidates must be positive.", nameof(nCandidates));

        // Draw order is fixed (Sobol seed, then the perturbation mask) so a per-trial seed reproduces the
        // whole candidate set. The Thompson z-draw (in ThompsonArgmax) continues on the same stream.
        int sobolSeed = (int)rng.randint(0, int.MaxValue);
        double[,] sobol = ScrambledSobol.Random(d, nCandidates, sobolSeed);   // [nCandidates, d] in [0,1)

        double probPerturb = Math.Min(20.0 / d, 1.0);
        double[] maskDraws = rng.uniform(0.0, 1.0, new Shape(nCandidates * d)).ToArray<double>();

        var candidates = new double[nCandidates, d];
        for (int i = 0; i < nCandidates; i++)
        {
            bool anyPerturbed = false;
            for (int j = 0; j < d; j++)
            {
                bool perturb = maskDraws[i * d + j] <= probPerturb;
                if (perturb)
                {
                    candidates[i, j] = lb[j] + (ub[j] - lb[j]) * sobol[i, j];
                    anyPerturbed = true;
                }
                else
                {
                    candidates[i, j] = xCenter[j];
                }
            }
            // The tutorial forces at least one perturbed coordinate so no candidate is a pure copy of the
            // incumbent (which would waste a candidate and cannot improve).
            if (!anyPerturbed)
            {
                int jr = (int)rng.randint(0, d);
                candidates[i, jr] = lb[jr] + (ub[jr] - lb[jr]) * sobol[i, jr];
            }
        }
        return candidates;
    }

    /// <summary>
    ///     Thompson sampling over a discrete candidate set: draw one realization of the GP posterior
    ///     function, <c>f = mean + L·z</c> with <c>cov = L·Lᵀ</c> and <c>z ~ N(0, I)</c>, and return the
    ///     index of the candidate with the largest sampled value (best maximization score). A small jitter
    ///     is escalated on the covariance diagonal to make the Cholesky robust to a numerically
    ///     near-singular joint posterior (candidates in a tight TR are highly correlated); if every jitter
    ///     level still fails, it falls back to the greedy argmax of the posterior mean.
    /// </summary>
    /// <param name="mean">The joint posterior mean at the candidates (maximization-oriented), length n.</param>
    /// <param name="cov">The joint posterior covariance at the candidates, <c>n × n</c> (symmetric PSD).</param>
    /// <param name="rng">The per-trial MT19937 stream (continues after the candidate generation).</param>
    /// <returns>The index of the argmax of the Thompson sample.</returns>
    /// <exception cref="System.ArgumentException">The covariance is not square or its size differs from the mean length.</exception>
    internal static int ThompsonArgmax(double[] mean, double[,] cov, NumPyRandom rng)
    {
        int n = mean.Length;
        if (cov.GetLength(0) != n || cov.GetLength(1) != n)
            throw new ArgumentException("cov must be an n×n matrix matching mean.");

        // z is drawn once (before the jitter loop) so the sample is identical regardless of how many jitter
        // levels the Cholesky needs — keeping the result reproducible from the per-trial seed.
        double[] z = NumPyDraws.Randn(rng, n);

        double[]? sample = null;
        foreach (double jitter in JitterLevels)
        {
            try
            {
                var c = (double[,])cov.Clone();
                for (int i = 0; i < n; i++) c[i, i] += jitter;
                double[,] l = LinAlg.Cholesky(c);   // c = L·Lᵀ
                double[] lz = LinAlg.Dot(l, z);     // L·z
                sample = new double[n];
                for (int i = 0; i < n; i++) sample[i] = mean[i] + lz[i];
                break;
            }
            catch (NumSharp.LinAlgError)
            {
                // The jittered covariance was still not positive definite; escalate the jitter and retry.
            }
        }

        // Fall back to pure exploitation (the posterior mean) if no jitter level produced a valid factor.
        double[] scores = sample ?? mean;
        int best = 0;
        double bestVal = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
            if (scores[i] > bestVal) { bestVal = scores[i]; best = i; }
        return best;
    }

    /// <summary>Escalating absolute jitter added to the covariance diagonal before Cholesky (standardized
    ///     score units are ~O(1), so an absolute ladder is appropriate).</summary>
    private static readonly double[] JitterLevels = { 1e-10, 1e-8, 1e-6, 1e-4, 1e-2 };

    /// <summary>Tests whether a scalar excludes NaN and both infinities.</summary>
    /// <param name="v">The scalar operand.</param>
    /// <returns>True for a finite scalar, including zero.</returns>
    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
}

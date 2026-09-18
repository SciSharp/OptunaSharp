// TuRBO-m — the multi-trust-region bandit layer over TuRBO-1 (Eriksson et al., NeurIPS 2019, §2 + §D/§E).
//
// Reference:
//   David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan D. Turner, Matthias Poloczek.
//     "Scalable Global Optimization via Local Bayesian Optimization." NeurIPS 2019 (arXiv:1910.01739).
//   Cross-region competition modeled on the authors' reference code (uber-research/TuRBO, turbo_m.py:
//   `_select_candidates`, `_create_candidates`) — the paper's §2 argmin/argmax over a Thompson sample drawn
//   independently from every region's local GP.
//
// This file holds the ONE mechanic that TuRBO-m adds over TuRBO-1: the implicit multi-armed bandit that
// picks which trust region spends the next evaluation. Everything INSIDE a region is unchanged TuRBO-1
// (`TurboOptimizer`/`TurboGpFit`, reused verbatim, one instance shared across regions). The paper (§2):
//   x_i ∈ argmin_ℓ argmin_{x∈TR_ℓ} f_ℓ^(i),  f_ℓ^(i) ∼ GP_ℓ(µ_ℓ, k_ℓ)
// i.e. draw one posterior realization from each region's GP over that region's candidate set, then take the
// single global optimum across all regions and candidates. Our scores are maximization-oriented, so this is
// a global ARGMAX. The subtle, load-bearing detail (turbo_m.py `y_cand = mu + sigma * y_cand`): every region
// standardizes its own objective values by its OWN mean/std before fitting, so each region's sample must be
// UN-STANDARDIZED back to shared objective units before the cross-region comparison — otherwise a strong
// region and a weak region are compared on incompatible z-scales. RNG is NumSharp's MT19937 (never
// System.Random); the caller consumes the per-trial stream in a fixed region order so the competition
// reproduces bit-for-bit from (seed, region labels).

using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Turbo;

/// <summary>
///     One ready trust region's payload for the cross-region Thompson competition (a value type so the
///     region list stays allocation-cheap). It carries the region's candidate set together with the
///     <b>standardized</b> joint GP posterior over those candidates and the region's own standardization
///     constants, which the competition uses to un-standardize each region's Thompson draw back to shared
///     maximization-oriented objective units before comparing regions.
///
///     <para><b>Footgun.</b> <see cref="CandidateMean" />/<see cref="CandidateCov" /> are in the region's
///     <i>standardized</i> units (the units the local GP was fitted in); comparing them across regions
///     directly is meaningless. <see cref="TurboMOptimizer.SelectAcrossTrustRegions" /> applies
///     <c>ScoreMean + ScoreStd·sample</c> per entry first — do not compare raw samples across regions.</para>
/// </summary>
internal readonly struct TrustRegionSample
{
    /// <summary>The region's slot index (<c>0 … m−1</c>) — the label persisted on the winning trial.</summary>
    public int Slot { get; }

    /// <summary>The region's normalized candidate points, <c>numCandidates × d</c> in <c>[0, 1]^d</c>
    ///     (from <see cref="TurboOptimizer.GenerateCandidates" />); the competition returns an index into these rows.</summary>
    public double[,] Candidates { get; }

    /// <summary>The joint GP posterior <b>mean</b> at the candidates, in the region's standardized units, length <c>numCandidates</c>.</summary>
    public double[] CandidateMean { get; }

    /// <summary>The joint GP posterior <b>covariance</b> at the candidates, in the region's standardized units, <c>numCandidates × numCandidates</c>.</summary>
    public double[,] CandidateCov { get; }

    /// <summary>The region's score-standardization center (the mean of its current run's maximization-oriented
    ///     scores) — added back when un-standardizing a Thompson draw for the cross-region comparison.</summary>
    public double ScoreMean { get; }

    /// <summary>The region's score-standardization scale (the std of its current run's scores, floored positive)
    ///     — multiplies the standardized draw when un-standardizing for the cross-region comparison.</summary>
    public double ScoreStd { get; }

    /// <summary>Captures one ready region's competition payload.</summary>
    /// <param name="slot">The region's slot index (<c>0 … m−1</c>).</param>
    /// <param name="candidates">The region's normalized candidate rows, <c>numCandidates × d</c>.</param>
    /// <param name="candidateMean">The standardized joint posterior mean at the candidates, length <c>numCandidates</c>.</param>
    /// <param name="candidateCov">The standardized joint posterior covariance at the candidates, <c>numCandidates × numCandidates</c>.</param>
    /// <param name="scoreMean">The region's score-standardization center (maximization-oriented).</param>
    /// <param name="scoreStd">The region's score-standardization scale (already floored positive).</param>
    public TrustRegionSample(
        int slot, double[,] candidates, double[] candidateMean, double[,] candidateCov,
        double scoreMean, double scoreStd)
    {
        Slot = slot;
        Candidates = candidates;
        CandidateMean = candidateMean;
        CandidateCov = candidateCov;
        ScoreMean = scoreMean;
        ScoreStd = scoreStd;
    }
}

/// <summary>
///     The multi-trust-region bandit of TuRBO-m (Eriksson et al. 2019, §2): the cross-region Thompson
///     competition and the deterministic init-fill schedule that keeps all <c>m</c> regions live. Everything
///     per-region is unchanged TuRBO-1 (<see cref="TurboOptimizer" />); this type adds only the "which region
///     spends the next evaluation" decision that the paper frames as an implicit multi-armed bandit.
/// </summary>
internal static class TurboMOptimizer
{
    /// <summary>Escalating absolute jitter added to a region's covariance diagonal before Cholesky — a verbatim
    ///     clone of <see cref="TurboOptimizer" />'s ladder so a region's Thompson draw here is numerically the
    ///     same one TuRBO-1 makes (cloned rather than shared to leave the single-region path provably untouched).</summary>
    private static readonly double[] JitterLevels = { 1e-10, 1e-8, 1e-6, 1e-4, 1e-2 };

    /// <summary>
    ///     The paper's cross-region selection for one candidate slot (§2, turbo_m.py <c>_select_candidates</c>):
    ///     draw one joint posterior realization from <b>each</b> region's local GP over that region's candidate
    ///     set, un-standardize every draw to shared maximization-oriented objective units, and return the single
    ///     globally best <c>(region, candidate)</c>. Whichever region's draw looks best wins the evaluation, so
    ///     promising regions naturally win more often — that emergent allocation <i>is</i> the bandit (there is
    ///     no separate arm-selection rule).
    ///
    ///     <para>Ties resolve to the first region (in list order) and then the first candidate, matching numpy's
    ///     first-occurrence <c>argmin</c> in the reference. Each region's <c>z</c> is drawn on the shared
    ///     <paramref name="rng" /> stream in list order, so with the regions supplied in a fixed slot order the
    ///     whole competition reproduces bit-for-bit from <c>(seed, region labels)</c>.</para>
    /// </summary>
    /// <param name="regions">The ready regions' payloads, supplied in a fixed slot order; each is sampled once, in that order.</param>
    /// <param name="rng">The per-trial MT19937 stream (already advanced by the per-region candidate generation).</param>
    /// <returns>The index (into <paramref name="regions" />) of the winning region and the winning candidate row within it.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="regions" /> is empty (there is nothing to compete).</exception>
    internal static (int RegionIndex, int Candidate) SelectAcrossTrustRegions(
        IReadOnlyList<TrustRegionSample> regions, NumPyRandom rng)
    {
        if (regions.Count == 0)
            throw new ArgumentException("At least one ready trust region is required to compete.", nameof(regions));

        int bestRegion = -1, bestCand = -1;
        double bestScore = double.NegativeInfinity;
        for (int r = 0; r < regions.Count; r++)
        {
            TrustRegionSample region = regions[r];
            int n = region.CandidateMean.Length;

            // Draw this region's independent Thompson realization on the shared stream, in region order, so the
            // outcome is a pure function of (seed, labels) regardless of how the host schedules trials.
            double[] z = rng.randn(n).ToArray<double>();
            double[] sample = ThompsonSample(region.CandidateMean, region.CandidateCov, z);

            for (int i = 0; i < n; i++)
            {
                // Un-standardize into shared objective units: each region standardized by its OWN mean/std, so a
                // raw z-score comparison across regions would confound "good region" with "low-variance region".
                double score = region.ScoreMean + region.ScoreStd * sample[i];
                // Strict '>' keeps the FIRST maximizer on ties (region-major, then candidate) — numpy argmin's
                // first-occurrence rule in maximization form.
                if (score > bestScore) { bestScore = score; bestRegion = r; bestCand = i; }
            }
        }
        return (bestRegion, bestCand);
    }

    /// <summary>
    ///     The deterministic init-fill decision (turbo_m.py's "initialize each TR independently", applied
    ///     sequentially): the lowest-indexed region slot that has not yet completed its current run's <c>nInit</c>
    ///     Sobol initialization, or <c>−1</c> when every region is ready to compete. Filling the lowest such slot
    ///     each trial fills the regions slot-by-slot at startup and forces a collapsed region's fresh <c>nInit</c>
    ///     re-init burst before the competition resumes, keeping all <c>m</c> regions live.
    /// </summary>
    /// <param name="trialsInCurrentRun">Per-slot count of completed trials in each region's <i>current</i> run
    ///     (after its most recent restart), length <c>m</c>.</param>
    /// <param name="nInit">The per-region initialization budget (Sobol points before the local GP participates).</param>
    /// <returns>The lowest-indexed slot still initializing, or <c>−1</c> if all slots are ready to compete.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="trialsInCurrentRun" /> is empty.</exception>
    internal static int NextInitSlot(int[] trialsInCurrentRun, int nInit)
    {
        if (trialsInCurrentRun.Length == 0)
            throw new ArgumentException("At least one trust region is required.", nameof(trialsInCurrentRun));
        for (int s = 0; s < trialsInCurrentRun.Length; s++)
            if (trialsInCurrentRun[s] < nInit)
                return s;
        return -1;
    }

    /// <summary>
    ///     Draws one joint GP-posterior realization over a region's candidate set, <c>f = mean + L·z</c> with
    ///     <c>cov = L·Lᵀ</c> — the §E Thompson sample, returning the whole sampled vector (the cross-region
    ///     competition needs every candidate's value, not just the argmax). Escalating diagonal jitter makes the
    ///     Cholesky robust to a near-singular joint posterior (tightly-correlated candidates in a small TR); if
    ///     every jitter level still fails it falls back to the posterior <paramref name="mean" /> (pure
    ///     exploitation). This is a verbatim clone of <see cref="TurboOptimizer.ThompsonArgmax" />'s draw so the
    ///     single-region path is left byte-for-byte untouched while the multi-region path samples identically.
    /// </summary>
    /// <param name="mean">The standardized joint posterior mean at the candidates, length n.</param>
    /// <param name="cov">The standardized joint posterior covariance at the candidates, <c>n × n</c> (symmetric PSD).</param>
    /// <param name="z">The standard-normal draw <c>z ~ N(0, I)</c>, length n (drawn once by the caller so the sample
    ///     is identical regardless of how many jitter levels the Cholesky needs).</param>
    /// <returns>The sampled vector <c>mean + L·z</c> (or <paramref name="mean" /> itself if no jitter level factored).</returns>
    private static double[] ThompsonSample(double[] mean, double[,] cov, double[] z)
    {
        int n = mean.Length;
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
                // Still not positive definite at this jitter level; escalate and retry.
            }
        }

        // Fall back to the posterior mean (pure exploitation) when no jitter level produced a valid factor.
        return sample ?? mean;
    }
}

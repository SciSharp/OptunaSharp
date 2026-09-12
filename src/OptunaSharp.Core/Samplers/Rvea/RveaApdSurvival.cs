// RVEA environmental survival — the Angle-Penalized Distance (APD) selection and the periodic
// reference-vector adaptation.
//
// Reference: Ran Cheng, Yaochu Jin, Markus Olhofer & Bernhard Sendhoff, "A Reference Vector Guided
// Evolutionary Algorithm for Many-Objective Optimization", IEEE TEVC 20(5), 2016. The APD metric,
// niche assignment and reference-vector adaptation follow pymoo's `APDSurvival`
// (pymoo/algorithms/moo/rvea.py, Apache-2.0). RVEA has NO Optuna/OptunaHub counterpart → native port
// validated by convergence/hypervolume, not bit-matched to an upstream.

using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Rvea;

/// <summary>
///     The Angle-Penalized-Distance survival operator of <see cref="RveaSampler" />. Each individual is
///     translated to make the running ideal point the origin, assigned to its nearest reference vector
///     (smallest acute angle), and within each reference vector the survivor is the individual with the
///     smallest APD — its distance to the ideal point scaled by a generation-growing angle penalty
///     <c>1 + M·(gen/gen_max)^α·(θ/γ)</c>. The reference vectors are periodically re-scaled to the current
///     objective ranges (<see cref="Adapt" />), which is what lets RVEA track differently-scaled objectives.
/// </summary>
internal sealed class RveaApdSurvival
{
    private readonly double _alpha;
    private readonly double _adaptFreq;
    private readonly int _nMaxGenerations;

    private readonly double[][] _v0;   // original unit reference vectors (n_ref × m) — the adaptation base
    private double[][] _v;             // current (possibly adapted) unit reference vectors
    private double[] _gamma;           // per-vector smallest angle to any other vector
    private double[] _ideal;           // running ideal point z* (minimization convention, length m)
    private double[]? _nadir;          // last survival's nadir over survivors (drives adaptation)
    private int _lastAdaptGeneration = -1;

    /// <param name="referenceVectors">Reference directions (n_ref × m), not necessarily unit length — they
    ///     are normalized here.</param>
    /// <param name="alpha">The APD penalty exponent α (Cheng et al.'s rate of the diversity→convergence shift). Default 2.</param>
    /// <param name="adaptFreq">Fraction of the maximum generation at which the reference vectors are re-scaled
    ///     (0 disables adaptation).</param>
    /// <param name="nMaxGenerations">The generation horizon used by the penalty schedule and the adaptation
    ///     period (Optuna is trial-budget-driven and does not expose a maximum generation, so it is set
    ///     explicitly).</param>
    /// <summary>Initializes reference vectors and the progress schedule for angle-penalized survival.</summary>
    internal RveaApdSurvival(double[][] referenceVectors, double alpha, double adaptFreq, int nMaxGenerations)
    {
        int m = referenceVectors[0].Length;
        _alpha = alpha;
        _adaptFreq = adaptFreq;
        _nMaxGenerations = Math.Max(1, nMaxGenerations);

        _v0 = NormalizeRows(referenceVectors);
        _v = _v0.Select(row => (double[])row.Clone()).ToArray();
        _gamma = CalcGamma(_v);
        _ideal = new double[m];
        for (int j = 0; j < m; j++) _ideal[j] = double.PositiveInfinity;
        _nadir = null;
    }

    /// <summary>The number of reference vectors (RVEA's effective population size cap).</summary>
    internal int RefCount => _v.Length;

    /// <summary>
    ///     Select the survivors of the combined population at generation <paramref name="generation" /> — at
    ///     most one per reference vector. Individuals with a non-finite objective are dropped (they cannot be
    ///     placed on the unit sphere); if that leaves nothing, the population is returned unchanged.
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <param name="generation">Zero-based generation whose individuals or optimizer state are requested.</param>
    /// <returns>The individuals retained by the configured survival rule, in selection order.</returns>
    internal List<FrozenTrial> Select(Study study, List<FrozenTrial> population, int generation)
    {
        var directions = study.Directions;
        int m = directions.Count;

        // Periodic reference-vector adaptation (pymoo adapts every ceil(n_max_gen * adapt_freq) generations).
        int period = Math.Max(1, (int)Math.Ceiling(_nMaxGenerations * _adaptFreq));
        if (_adaptFreq > 0.0 && generation > 0 && generation % period == 0
            && generation != _lastAdaptGeneration && _nadir is not null)
        {
            Adapt();
            _lastAdaptGeneration = generation;
        }

        // Loss matrix in the minimization convention; drop individuals with a non-finite objective.
        var feasible = new List<FrozenTrial>(population.Count);
        var lossList = new List<double[]>(population.Count);
        foreach (var t in population)
        {
            var loss = new double[m];
            bool finite = true;
            for (int j = 0; j < m; j++)
            {
                loss[j] = MultiObjective.NormalizeValue(t.Values![j], directions[j]);
                if (!double.IsFinite(loss[j])) { finite = false; break; }
            }
            if (!finite) continue;
            feasible.Add(t);
            lossList.Add(loss);
        }
        if (feasible.Count == 0)
            return population;

        double[][] f = lossList.ToArray();
        int n = f.Length;

        // Running ideal point on the original scale, then translate so the ideal is the origin.
        for (int j = 0; j < m; j++)
        {
            double colMin = double.PositiveInfinity;
            for (int i = 0; i < n; i++) if (f[i][j] < colMin) colMin = f[i][j];
            if (colMin < _ideal[j]) _ideal[j] = colMin;
        }

        var translated = new double[n][];
        var distToIdeal = new double[n];
        for (int i = 0; i < n; i++)
        {
            translated[i] = new double[m];
            double s = 0.0;
            for (int j = 0; j < m; j++)
            {
                double v = f[i][j] - _ideal[j];
                translated[i][j] = v;
                s += v * v;
            }
            distToIdeal[i] = Math.Max(Math.Sqrt(s), 1e-64);
        }

        // Acute angle of each individual to each reference vector; niche = nearest reference vector.
        int refCount = _v.Length;
        var acuteAngle = new double[n][];
        var niche = new int[n];
        for (int i = 0; i < n; i++)
        {
            acuteAngle[i] = new double[refCount];
            int best = 0;
            double bestAngle = double.PositiveInfinity;
            for (int k = 0; k < refCount; k++)
            {
                double dot = 0.0;
                for (int j = 0; j < m; j++) dot += (translated[i][j] / distToIdeal[i]) * _v[k][j];
                double angle = Math.Acos(Math.Min(1.0, Math.Max(-1.0, dot)));
                acuteAngle[i][k] = angle;
                if (angle < bestAngle) { bestAngle = angle; best = k; }
            }
            niche[i] = best;
        }

        var nichesToInd = new List<int>[refCount];
        for (int k = 0; k < refCount; k++) nichesToInd[k] = new List<int>();
        for (int i = 0; i < n; i++) nichesToInd[niche[i]].Add(i);

        double mCoef = m > 2 ? m : 1.0;
        double genRatio = Math.Min(1.0, generation / (double)_nMaxGenerations);   // clamp to [0,1]
        double ratioAlpha = Math.Pow(genRatio, _alpha);

        var survivors = new List<int>();
        for (int k = 0; k < refCount; k++)
        {
            var assigned = nichesToInd[k];
            if (assigned.Count == 0) continue;

            double gamma = _gamma[k];
            int survivor = assigned[0];
            double bestApd = double.PositiveInfinity;
            foreach (int i in assigned)
            {
                double theta = acuteAngle[i][k];
                double penalty = mCoef * ratioAlpha * (theta / gamma);
                double apd = distToIdeal[i] * (1.0 + penalty);
                if (apd < bestApd) { bestApd = apd; survivor = i; }
            }
            survivors.Add(survivor);
        }

        // Nadir over the survivors (original scale) — the range estimate the next adaptation uses.
        var nadir = new double[m];
        for (int j = 0; j < m; j++) nadir[j] = double.NegativeInfinity;
        foreach (int i in survivors)
            for (int j = 0; j < m; j++)
                if (f[i][j] > nadir[j]) nadir[j] = f[i][j];
        _nadir = nadir;

        return survivors.Select(i => feasible[i]).ToList();
    }

    /// <summary>pymoo <c>adapt</c>: rescale each reference vector by the current objective range
    ///     <c>(nadir − ideal)</c> and re-normalize (a zero-range axis leaves that vector unchanged).</summary>
    private void Adapt()
    {
        int m = _v0[0].Length;
        var range = new double[m];
        for (int j = 0; j < m; j++) range[j] = _nadir![j] - _ideal[j];

        var scaled = new double[_v0.Length][];
        for (int k = 0; k < _v0.Length; k++)
        {
            var row = new double[m];
            for (int j = 0; j < m; j++) row[j] = _v0[k][j] * range[j];
            scaled[k] = row;
        }
        _v = NormalizeRows(scaled, fallback: _v0);
        _gamma = CalcGamma(_v);
    }

    /// <summary>calc_V: normalize each row to unit length. A zero row is replaced by the corresponding
    ///     <paramref name="fallback" /> row (or left zero when no fallback is given).</summary>
    /// <param name="rows">Observation rows to copy into the result.</param>
    /// <param name="fallback">Optional unit-vector rows substituted for degenerate directions.</param>
    /// <returns>Unit-length reference vectors, with configured fallback rows for degenerate inputs.</returns>
    private static double[][] NormalizeRows(double[][] rows, double[][]? fallback = null)
    {
        var result = new double[rows.Length][];
        for (int k = 0; k < rows.Length; k++)
        {
            double norm = 0.0;
            for (int j = 0; j < rows[k].Length; j++) norm += rows[k][j] * rows[k][j];
            norm = Math.Sqrt(norm);
            if (norm <= 0.0)
            {
                result[k] = fallback is not null ? (double[])fallback[k].Clone() : (double[])rows[k].Clone();
                continue;
            }
            var row = new double[rows[k].Length];
            for (int j = 0; j < rows[k].Length; j++) row[j] = rows[k][j] / norm;
            result[k] = row;
        }
        return result;
    }

    /// <summary>calc_gamma: for each reference vector, the smallest angle to any other reference vector
    ///     (pymoo: <c>arccos(-sort(-V@V.T)[:,1])</c>, floored at 1e-64).</summary>
    /// <param name="v">Vector operand aligned with the matrix coordinates.</param>
    /// <returns>The smallest inter-vector angle for each reference vector.</returns>
    private static double[] CalcGamma(double[][] v)
    {
        int refCount = v.Length;
        int m = v[0].Length;
        var gamma = new double[refCount];
        for (int k = 0; k < refCount; k++)
        {
            // Largest cosine to another vector = smallest angle (pymoo's (-sort(-V@V.T))[:,1], since the
            // self-cosine 1.0 sits at index 0). A single vector has no neighbor ⇒ default to 90°.
            double maxOtherCos = double.NegativeInfinity;
            for (int l = 0; l < refCount; l++)
            {
                if (l == k) continue;
                double dot = 0.0;
                for (int j = 0; j < m; j++) dot += v[k][j] * v[l][j];
                if (dot > maxOtherCos) maxOtherCos = dot;
            }
            double cos = refCount >= 2 ? maxOtherCos : 0.0;
            gamma[k] = Math.Max(Math.Acos(Math.Min(1.0, Math.Max(-1.0, cos))), 1e-64);
        }
        return gamma;
    }
}

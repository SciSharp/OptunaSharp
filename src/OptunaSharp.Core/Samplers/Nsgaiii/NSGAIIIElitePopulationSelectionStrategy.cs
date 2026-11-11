using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Samplers.Nsgaiii;

/// <summary>
///     The default elite-population selection strategy of <see cref="NSGAIIISampler" />. Port of
///     <c>optuna/samplers/_nsgaiii/_elite_population_selection_strategy.py</c>.
///
///     Ranks the combined population by (constrained) non-domination (shared
///     <see cref="NSGAIIElitePopulationSelectionStrategy.RankPopulation" />), then fills the elite
///     population rank by rank. The rank that overflows <c>population_size</c> — the borderline front —
///     is truncated by <b>reference-point niching</b> rather than crowding distance: objective values
///     are normalized onto a hyperplane (ideal point + achievement-scalarizing extreme points), each
///     individual is associated with its closest reference point, and survivors are drawn to keep the
///     reference points as evenly populated (sparse) as possible.
///
///     <para><b>Bit-exact.</b> The whole pipeline matches Optuna bit-for-bit: <c>_filter_inf</c>,
///     the ideal-point subtraction, the ASF <c>einsum</c>/argmin extreme-point pick, the
///     reference-point association (numpy's 4-operand <c>einsum</c> replicated term-by-term, verified
///     bit-identical over 2000 cases; its two <c>np.linalg.norm(axis=...)</c> calls and the axis
///     <c>argmin</c>/<c>min</c> run through NumSharp, so the row sums take NumPy's pairwise order from
///     eight objectives on — before 2026-11-11 they were left-to-right sums, exact only up to seven
///     objectives), the niching, and — since 2.15 — the hyperplane fit itself: the
///     degeneracy <see cref="LinAlg.MatrixRank" /> branch and the <see cref="LinAlg.Solve" /> route
///     through NumSharp's <c>np.linalg.matrix_rank</c>/<c>solve</c> over the OpenBLAS LAPACK backend,
///     which is byte-identical to NumPy (the earlier managed reciprocal-multiplier LU drifted ~1 ULP
///     on a few percent of the extreme-point systems; the LAPACK route removes even that). Every RNG
///     draw (<c>rng.shuffle</c>) stays in lock-step. Confirmed byte-for-byte over a 400-config
///     differential (5 crossovers × {2,3,4} objectives × 25 seeds + 25 constrained); the many-objective
///     regime (8 and 10 objectives) is pinned by the live
///     <c>NsgaiiiSamplerParityTests.ManyObjectives_FullState_IsBitIdentical</c>.</para>
/// </summary>
public class NSGAIIIElitePopulationSelectionStrategy
{
    // Define a coefficient for scaling intervals, used in FilterInf to replace +-inf.
    private const int Coef = 3;

    private readonly int _populationSize;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private double[,]? _referencePoints;
    private readonly int _dividingParameter;
    private readonly LazyRandomState _rng;

    /// <summary>Configures nondomination ranking and reference-point niching to a fixed population size.</summary>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="constraintsFunc">Optional legacy callback returning constraint values; positive values violate a constraint.</param>
    /// <param name="referencePoints">Reference vectors, one row per point and one column per objective.</param>
    /// <param name="dividingParameter">Number of simplex divisions used to generate reference points.</param>
    /// <exception cref="NumSharp.ValueError">Population size is below two.</exception>
    public NSGAIIIElitePopulationSelectionStrategy(
        int populationSize,
        LazyRandomState rng,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        double[,]? referencePoints = null,
        int dividingParameter = 3)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        _populationSize = populationSize;
        _constraintsFunc = constraintsFunc;
        _referencePoints = referencePoints;
        _dividingParameter = dividingParameter;
        _rng = rng;
    }

    /// <summary>
    ///     Select the elite population from the given trials by the NSGA-III algorithm (Python
    ///     <c>__call__</c>).
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <returns>The individuals retained by the configured survival rule, in selection order.</returns>
    /// <exception cref="NumSharp.ValueError">The reference-point dimension does not match the study objective count.</exception>
    public List<FrozenTrial> Select(Study study, List<FrozenTrial> population)
    {
        bool isConstrained = ConstrainedOptimization.IsConstrainedOptimization(population);
        NsgaiiConstraintsEvaluation.ValidateConstraints(population, isConstrained);
        var populationPerRank = NSGAIIElitePopulationSelectionStrategy.RankPopulation(
            population, study.Directions, isConstrained);

        var elitePopulation = new List<FrozenTrial>();
        foreach (var individuals in populationPerRank)
        {
            if (elitePopulation.Count + individuals.Count < _populationSize)
            {
                elitePopulation.AddRange(individuals);
            }
            else
            {
                int nObjectives = study.Directions.Count;
                // Construct reference points in the first run.
                if (_referencePoints is null)
                    _referencePoints = GenerateDefaultReferencePoint(nObjectives, _dividingParameter);
                else if (_referencePoints.GetLength(1) != nObjectives)
                    throw new ValueError(
                        "The dimension of reference points vectors must be the same as the number " +
                        "of objectives of the study.");

                // Normalize objective values after filtering +-inf.
                var combined = new List<FrozenTrial>(elitePopulation);
                combined.AddRange(individuals);
                double[,] objectiveMatrix = NormalizeObjectiveValues(FilterInf(combined));

                var (closestReferencePoints, distanceReferencePoints) =
                    AssociateIndividualsWithReferencePoints(objectiveMatrix, _referencePoints);

                int elitePopulationNum = elitePopulation.Count;
                int targetPopulationSize = _populationSize - elitePopulationNum;
                var additionalElitePopulation = PreserveNicheIndividuals(
                    targetPopulationSize, elitePopulationNum, individuals,
                    closestReferencePoints, distanceReferencePoints);
                elitePopulation.AddRange(additionalElitePopulation);
                break;
            }
        }

        return elitePopulation;
    }

    /// <summary>
    ///     Python <c>_generate_default_reference_point</c>: the reference points `uniformly` spread on
    ///     a hyperplane, one per <c>combinations_with_replacement(range(n_objectives), div)</c>.
    /// </summary>
    /// <param name="nObjectives">Number of objective coordinates.</param>
    /// <param name="dividingParameter">Number of simplex divisions used to generate reference points.</param>
    /// <returns>Simplex reference points with one row per point and one column per objective.</returns>
    internal static double[,] GenerateDefaultReferencePoint(int nObjectives, int dividingParameter = 3)
    {
        var indices = CombinationsWithReplacement(nObjectives, dividingParameter);
        var referencePoints = new double[indices.Count, nObjectives];
        // np.add.at(reference_points, (row, col), 1.0): each combination contributes +1 per index it
        // lists (with multiplicity), so a point can have entries > 1 (e.g. (0,0,0) -> [3, 0, ...]).
        for (int r = 0; r < indices.Count; r++)
            foreach (int col in indices[r])
                referencePoints[r, col] += 1.0;
        return referencePoints;
    }

    /// <summary>
    ///     <c>itertools.combinations_with_replacement(range(n), r)</c> in lexicographic order — the
    ///     same order numpy sees, so the reference-point rows line up byte-for-byte.
    /// </summary>
    /// <param name="n">Number of distinct zero-based index values available.</param>
    /// <param name="r">Number of entries in each combination.</param>
    /// <returns>Nondecreasing index combinations of the requested length.</returns>
    private static List<int[]> CombinationsWithReplacement(int n, int r)
    {
        var result = new List<int[]>();
        if (n == 0)
            return result;
        var indices = new int[r];  // starts at all-zeros == (0, 0, ..., 0)
        while (true)
        {
            result.Add((int[])indices.Clone());
            // Advance like CPython's combinations_with_replacement: find rightmost index < n-1.
            int i = r - 1;
            while (i >= 0 && indices[i] == n - 1)
                i--;
            if (i < 0)
                break;
            int next = indices[i] + 1;
            for (int j = i; j < r; j++)
                indices[j] = next;
        }
        return result;
    }

    /// <summary>Python <c>_filter_inf</c>: clip +-inf to a margin beyond the finite range per axis.</summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <returns>A finite objective matrix using the reference replacement for infinite entries.</returns>
    private static double[,] FilterInf(IReadOnlyList<FrozenTrial> population)
    {
        int p = population.Count;
        int m = population[0].Values!.Count;

        // np.nanmax / np.nanmin over the finite entries of each column (inf -> nan first).
        var maxObjectives = new double[m];
        var minObjectives = new double[m];
        for (int j = 0; j < m; j++)
        {
            double cmax = double.NegativeInfinity, cmin = double.PositiveInfinity;
            bool seen = false;
            for (int i = 0; i < p; i++)
            {
                double v = population[i].Values![j];
                if (double.IsFinite(v))
                {
                    seen = true;
                    if (v > cmax) cmax = v;
                    if (v < cmin) cmin = v;
                }
            }
            maxObjectives[j] = seen ? cmax : double.NaN;
            minObjectives[j] = seen ? cmin : double.NaN;
        }

        var result = new double[p, m];
        for (int j = 0; j < m; j++)
        {
            double margin = Coef * (maxObjectives[j] - minObjectives[j]);
            double lo = minObjectives[j] - margin;
            double hi = maxObjectives[j] + margin;
            for (int i = 0; i < p; i++)
            {
                double v = population[i].Values![j];
                // np.clip(v, lo, hi) == min(max(v, lo), hi).
                result[i, j] = Math.Min(Math.Max(v, lo), hi);
            }
        }
        return result;
    }

    /// <summary>Python <c>_normalize_objective_values</c>. Mutates and returns <paramref name="a" />.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>Objective rows translated by the ideal point and scaled by intercepts.</returns>
    private static double[,] NormalizeObjectiveValues(double[,] a)
    {
        int p = a.GetLength(0);
        int m = a.GetLength(1);

        // Subtract ideal point (per-column min) from objective values.
        for (int j = 0; j < m; j++)
        {
            double cmin = a[0, j];
            for (int i = 1; i < p; i++)
                if (a[i, j] < cmin) cmin = a[i, j];
            for (int i = 0; i < p; i++)
                a[i, j] -= cmin;
        }

        // weights = np.eye(m); weights[weights == 0] = 1e6.
        var weights = new double[m, m];
        for (int d = 0; d < m; d++)
            for (int j = 0; j < m; j++)
                weights[d, j] = d == j ? 1.0 : 1e6;

        // asf[d, n] = max_j a[n, j] * weights[d, j]; extreme_points[d] = a[argmin_n asf[d, n]].
        // (einsum("nm,dm->dnm") is an elementwise product; max over m then argmin over n.)
        var extremePoints = new double[m, m];
        for (int d = 0; d < m; d++)
        {
            int bestN = 0;
            double bestAsf = double.NaN;
            bool first = true;
            for (int n = 0; n < p; n++)
            {
                double mx = double.NaN;
                bool firstM = true;
                for (int j = 0; j < m; j++)
                {
                    double val = a[n, j] * weights[d, j];
                    if (firstM || val > mx) { mx = val; firstM = false; }
                }
                if (first || mx < bestAsf) { bestAsf = mx; bestN = n; first = false; }
            }
            for (int j = 0; j < m; j++)
                extremePoints[d, j] = a[bestN, j];
        }

        // Normalize with extreme points, or fall back to per-axis max when they are degenerate.
        var interceptsInv = new double[m];
        bool allFinite = true;
        for (int d = 0; d < m && allFinite; d++)
            for (int j = 0; j < m; j++)
                if (!double.IsFinite(extremePoints[d, j])) { allFinite = false; break; }

        if (allFinite && LinAlg.MatrixRank(extremePoints) == m)
        {
            var ones = new double[m];
            for (int j = 0; j < m; j++) ones[j] = 1.0;
            interceptsInv = LinAlg.Solve(extremePoints, ones);
        }
        else
        {
            for (int j = 0; j < m; j++)
            {
                double intercept = a[0, j];
                for (int i = 1; i < p; i++)
                    if (a[i, j] > intercept) intercept = a[i, j];
                interceptsInv[j] = 1.0 / (intercept == 0.0 ? 1.0 : intercept);
            }
        }

        // objective_matrix *= np.where(np.isfinite(intercepts_inv), intercepts_inv, 1).
        for (int j = 0; j < m; j++)
        {
            double scale = double.IsFinite(interceptsInv[j]) ? interceptsInv[j] : 1.0;
            for (int i = 0; i < p; i++)
                a[i, j] *= scale;
        }

        return a;
    }

    /// <summary>
    ///     Python <c>_associate_individuals_with_reference_points</c>: the closest reference point to
    ///     each individual and that Euclidean (perpendicular) distance.
    /// </summary>
    /// <param name="objectiveMatrix">Normalized objective vectors with one individual per row.</param>
    /// <param name="referencePoints">Reference vectors, one row per point and one column per objective.</param>
    /// <returns>
    ///     For each individual, the index of the reference line closest to it and the distance to that line. A row
    ///     with a NaN distance reports the first NaN's line and NaN, as <c>np.argmin</c> / <c>np.min</c> do.
    /// </returns>
    /// <remarks>
    ///     <para>
    ///         Same statements as Optuna's, so the bits follow NumPy at every objective count. The two norms and the
    ///         axis-1 <c>argmin</c> / <c>min</c> are NumSharp's: a row norm over the last axis sums the squares in
    ///         NumPy's pairwise order, which from eight objectives on is not a left-to-right sum (the former loop
    ///         matched NumPy up to seven objectives and missed every distance from eight on, measured 2026-11-11
    ///         over 54 random cases), and the axis argmin / min let a NaN distance win the row as in NumPy, where a
    ///         strict <c>&lt;</c> loop skipped it. A 10-objective NSGA-III run (population 20, seed 0) left
    ///         Optuna's trajectory at trial 100 before this.
    ///     </para>
    ///     <para>
    ///         The 4-operand <c>einsum("ni,pi,p,pm->npm")</c> stays a hand port: NumPy's default <c>c_einsum</c>
    ///         accumulates <c>((a·b)·c)·d</c> over <c>i</c> term by term, which the loop reproduces bit for bit,
    ///         where NumSharp's <c>np.einsum</c> folds the operands pairwise through matrix products (the
    ///         <c>optimize=</c> path) and matched NumPy in none of the 54 cases. Cost: one (individuals × reference
    ///         points × objectives) tensor per generation, as in Optuna (about 15 ms at 200 individuals, 220
    ///         reference points and 10 objectives).
    ///     </para>
    ///     <para>
    ///         The caller (<see cref="Select" />) has already checked that the reference points have one column per
    ///         objective, with Optuna's <c>ValueError</c>.
    ///     </para>
    /// </remarks>
    private static (int[] Closest, double[] Distance) AssociateIndividualsWithReferencePoints(
        double[,] objectiveMatrix, double[,] referencePoints)
    {
        int p = objectiveMatrix.GetLength(0);
        int m = objectiveMatrix.GetLength(1);
        int r = referencePoints.GetLength(0);

        // reference_point_norm_squared = np.linalg.norm(reference_points, axis=1) ** 2, then its reciprocal. The norm is
        // rounded before it is squared (sqrt(s) ** 2 is not s in float64), as in Optuna.
        double[] inverseNormSquared;
        using (NDArray referenceArray = np.array(referencePoints))
        using (NDArray referenceNorms = np.linalg.norm(referenceArray, axis: 1))
            inverseNormSquared = referenceNorms.ToArray<double>();
        for (int q = 0; q < r; q++)
            inverseNormSquared[q] = 1.0 / (inverseNormSquared[q] * inverseNormSquared[q]);

        // objective_matrix[:, np.newaxis, :] - perpendicular_vectors_to_reference_lines, where the perpendicular vectors
        // are einsum("ni,pi,p,pm->npm", objective_matrix, reference_points, 1 / norm_squared, reference_points): each
        // element is c_einsum's left-to-right sum over i of ((obj[n,i] * ref[q,i]) * inv[q]) * ref[q,k].
        var differences = new double[p, r, m];
        for (int n = 0; n < p; n++)
        {
            for (int q = 0; q < r; q++)
            {
                for (int k = 0; k < m; k++)
                {
                    double perpendicular = 0.0;
                    for (int i = 0; i < m; i++)
                        perpendicular += objectiveMatrix[n, i] * referencePoints[q, i] * inverseNormSquared[q] * referencePoints[q, k];
                    differences[n, q, k] = objectiveMatrix[n, k] - perpendicular;
                }
            }
        }

        // distance_from_reference_lines = np.linalg.norm(differences, axis=2); then np.argmin and np.min along axis 1,
        // which keep the first minimum on ties and let the first NaN win a row.
        using NDArray differenceArray = np.array(differences);
        using NDArray distances = np.linalg.norm(differenceArray, axis: 2);
        using NDArray closestArray = np.argmin(distances, axis: 1);
        using NDArray distanceArray = np.amin(distances, axis: 1);
        long[] closestIndices = closestArray.ToArray<long>();
        var closest = new int[p];
        for (int n = 0; n < p; n++)
            closest[n] = (int)closestIndices[n];
        return (closest, distanceArray.ToArray<double>());
    }

    /// <summary>
    ///     Python <c>_preserve_niche_individuals</c>: draw survivors from the borderline front so the
    ///     reference points stay as evenly (sparsely) populated as possible.
    /// </summary>
    /// <param name="targetPopulationSize">Number of individuals to retain after selection.</param>
    /// <param name="elitePopulationNum">Number of individuals already retained from better fronts.</param>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <param name="closestReferencePoints">Nearest reference-point index per individual.</param>
    /// <param name="distanceReferencePoints">Perpendicular distance to the nearest reference point per individual.</param>
    /// <returns>Individuals chosen from the split front to fill underrepresented reference niches.</returns>
    /// <exception cref="NumSharp.ValueError">The candidate population contains fewer individuals than the target population size.</exception>
    private List<FrozenTrial> PreserveNicheIndividuals(
        int targetPopulationSize,
        int elitePopulationNum,
        List<FrozenTrial> population,
        int[] closestReferencePoints,
        double[] distanceReferencePoints)
    {
        if (population.Count < targetPopulationSize)
            throw new ValueError(
                "The population size must be greater than or equal to the target population size.");

        // reference_point -> list of (distance, borderline-local-index), in first-seen key order.
        var refToBorderline = new Dictionary<int, List<(double Distance, int Index)>>();
        var refInsertionOrder = new List<int>();
        for (int i = 0; i < closestReferencePoints.Length - elitePopulationNum; i++)
        {
            int refIdx = closestReferencePoints[i + elitePopulationNum];
            int populationIdx = i + elitePopulationNum;
            if (!refToBorderline.TryGetValue(refIdx, out var list))
            {
                list = new List<(double, int)>();
                refToBorderline[refIdx] = list;
                refInsertionOrder.Add(refIdx);
            }
            list.Add((distanceReferencePoints[populationIdx], i));
        }

        // reference_point -> how many elite neighbors it has (defaultdict(int)).
        var refToEliteCount = new Dictionary<int, int>();
        for (int i = 0; i < elitePopulationNum; i++)
        {
            int refIdx = closestReferencePoints[i];
            refToEliteCount[refIdx] = refToEliteCount.GetValueOrDefault(refIdx, 0) + 1;
        }

        // elite-neighbor-count -> reference points (only those with a borderline member), in
        // first-seen key order (defaultdict(list)).
        var countToRefs = new Dictionary<int, List<int>>();
        foreach (int refIdx in refInsertionOrder)
        {
            int eliteCount = refToEliteCount.GetValueOrDefault(refIdx, 0);
            GetBucket(countToRefs, eliteCount).Add(refIdx);
        }

        int count = -1;
        var additionalElitePopulation = new List<FrozenTrial>();
        var isShuffled = new HashSet<int>();
        while (additionalElitePopulation.Count < targetPopulationSize)
        {
            var currentBucket = GetBucket(countToRefs, count);
            if (currentBucket.Count == 0)
            {
                count += 1;
                ShuffleInPlace(GetBucket(countToRefs, count));
                continue;
            }

            int refIdx = currentBucket[^1];
            currentBucket.RemoveAt(currentBucket.Count - 1);

            var borderline = refToBorderline[refIdx];
            if (count > 0 && !isShuffled.Contains(refIdx))
            {
                ShuffleInPlace(borderline);
                isShuffled.Add(refIdx);
            }
            else if (count == 0)
            {
                // list.sort(reverse=True) on (distance, index) tuples (index is unique -> total order).
                borderline.Sort((x, y) =>
                {
                    int c = y.Distance.CompareTo(x.Distance);
                    return c != 0 ? c : y.Index.CompareTo(x.Index);
                });
            }

            var selected = borderline[^1];
            borderline.RemoveAt(borderline.Count - 1);
            additionalElitePopulation.Add(population[selected.Index]);
            if (borderline.Count > 0)
                GetBucket(countToRefs, count + 1).Add(refIdx);
        }

        return additionalElitePopulation;
    }

    /// <summary>Python <c>defaultdict(list)</c> access: auto-create the bucket for a missing key.</summary>
    /// <param name="dict">Dictionary mapping niche identifiers to mutable element lists.</param>
    /// <param name="key">Niche identifier whose element list is requested.</param>
    /// <typeparam name="T">Element type stored in the niche bucket or shuffled list.</typeparam>
    /// <returns>The existing list for the key, or a newly inserted empty list.</returns>
    private static List<T> GetBucket<T>(Dictionary<int, List<T>> dict, int key)
    {
        if (!dict.TryGetValue(key, out var list))
        {
            list = new List<T>();
            dict[key] = list;
        }
        return list;
    }

    /// <summary>
    ///     <c>rng.shuffle(list)</c> in place — reproduces numpy's Python-list Fisher-Yates by shuffling
    ///     an index permutation with the same MT19937 j-sequence (bit-exact, verified) and gathering.
    /// </summary>
    /// <param name="list">Elements reordered in place.</param>
    /// <typeparam name="T">Element type stored in the niche bucket or shuffled list.</typeparam>
    private void ShuffleInPlace<T>(List<T> list)
    {
        int n = list.Count;
        if (n <= 1)
            return;  // numpy draws no random numbers for length 0 or 1.
        var permutation = np.arange(n);
        _rng.Rng.shuffle(permutation);
        long[] perm = permutation.ToArray<long>();
        var copy = new List<T>(list);
        for (int k = 0; k < n; k++)
            list[k] = copy[(int)perm[k]];
    }
}

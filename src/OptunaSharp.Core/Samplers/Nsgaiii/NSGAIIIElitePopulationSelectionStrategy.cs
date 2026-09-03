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
///     <para><b>[Misaligned] managed-linalg tolerance (the hyperplane fit).</b>
///     <see cref="NsgaiiiLinAlg.Solve" /> is an unblocked reciprocal-multiplier LU (LAPACK
///     <c>dgetf2</c> shape) that reproduces numpy's <c>np.linalg.solve</c> bit-for-bit on the vast
///     majority of the small extreme-point systems that arise, but drifts by ~1 ULP on a few percent
///     of them (the managed-vs-LAPACK class the plan excuses — §10). Everything else in the pipeline
///     is bit-exact: <c>_filter_inf</c>, the ideal-point subtraction, the ASF <c>einsum</c>/argmin
///     extreme-point pick, the reference-point association (numpy's 4-operand <c>einsum</c> is
///     replicated term-by-term, verified bit-identical over 2000 cases), the degeneracy
///     <see cref="NsgaiiiLinAlg.MatrixRank" /> branch, and the niching itself. A 1-ULP normalization
///     difference only changes the outcome when it flips a niche-distance decision within ULPs of a
///     tie — otherwise a study's full trajectory matches Optuna bit-for-bit. Every RNG draw
///     (<c>rng.shuffle</c>) stays in lock-step regardless.</para>
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

        if (allFinite && NsgaiiiLinAlg.MatrixRank(extremePoints) == m)
        {
            var ones = new double[m];
            for (int j = 0; j < m; j++) ones[j] = 1.0;
            interceptsInv = NsgaiiiLinAlg.Solve(extremePoints, ones);
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
    private static (int[] Closest, double[] Distance) AssociateIndividualsWithReferencePoints(
        double[,] objectiveMatrix, double[,] referencePoints)
    {
        int p = objectiveMatrix.GetLength(0);
        int m = objectiveMatrix.GetLength(1);
        int r = referencePoints.GetLength(0);

        // 1 / (norm(reference_points, axis=1) ** 2) — note norm then square (sqrt(s)**2 != s exactly).
        var inv = new double[r];
        for (int q = 0; q < r; q++)
        {
            double s = 0.0;
            for (int j = 0; j < m; j++) s += referencePoints[q, j] * referencePoints[q, j];
            double t = Math.Sqrt(s);
            inv[q] = 1.0 / (t * t);
        }

        var closest = new int[p];
        var distance = new double[p];
        for (int n = 0; n < p; n++)
        {
            int best = 0;
            double bestDist = double.NaN;
            bool first = true;
            for (int q = 0; q < r; q++)
            {
                // dist = norm(obj_n - perp_{n,q}); perp_{n,q,k} matches numpy's 4-operand
                // einsum("ni,pi,p,pm->npm") accumulated term-by-term (bit-exact, verified).
                double s = 0.0;
                for (int k = 0; k < m; k++)
                {
                    double perp = 0.0;
                    for (int i = 0; i < m; i++)
                        perp += objectiveMatrix[n, i] * referencePoints[q, i] * inv[q] * referencePoints[q, k];
                    double diff = objectiveMatrix[n, k] - perp;
                    s += diff * diff;
                }
                double dist = Math.Sqrt(s);
                // np.argmin / np.min keep the FIRST minimum on ties (strict <).
                if (first || dist < bestDist) { bestDist = dist; best = q; first = false; }
            }
            closest[n] = best;
            distance[n] = bestDist;
        }

        return (closest, distance);
    }

    /// <summary>
    ///     Python <c>_preserve_niche_individuals</c>: draw survivors from the borderline front so the
    ///     reference points stay as evenly (sparsely) populated as possible.
    /// </summary>
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

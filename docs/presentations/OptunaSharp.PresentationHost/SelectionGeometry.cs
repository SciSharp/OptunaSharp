using NumSharp;

namespace OptunaSharp.PresentationHost;

/// <summary>
/// Presentation-only geometry reconstructed from finite minimization objectives. This mirrors
/// the core's deterministic normalization and association, but never selects a survivor or draws
/// random numbers. Survivor IDs and their niching order come from the real sampler return value.
/// Keep in sync with Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs.
/// </summary>
public static class SelectionGeometry
{
    // The core stores integer lattice points, whose rays are identical to these simplex points.
    // Preserve core combinations-with-replacement order for matching reference indices.
    public static double[][] References(int divisions)
    {
        if (divisions is < 1 or > 16)
            throw new ArgumentException("Reference divisions must be between 1 and 16.");
        var result = new List<double[]>();
        for (var a = divisions; a >= 0; a--)
            for (var b = divisions - a; b >= 0; b--)
                result.Add([(double)a / divisions, (double)b / divisions,
                    (double)(divisions - a - b) / divisions]);
        return result.ToArray();
    }

    public static ReferenceLabResult ReferenceLab(ReferenceLabRequest request)
    {
        var refs = References(request.Divisions);
        var point = request.Point ?? [0.45, 0.35, 0.20];
        if (point.Length != 3 || point.Any(v => !double.IsFinite(v) || v is < 0 or > 5))
            throw new ArgumentException("Point must contain three finite coordinates between 0 and 5.");
        var association = Associate(point, refs);
        return new ReferenceLabResult(refs, point, association.Index,
            association.Distance, association.Projection);
    }

    public static RunSnapshot Snapshot(string id, int generation, RunConfig config,
        List<FrozenTrial> parents, List<FrozenTrial> offspring, List<FrozenTrial> selected)
    {
        // This is the core's candidate ordering, relevant to deterministic distance ties.
        var combined = offspring.Concat(parents).ToArray();
        var fronts = Rank(combined);
        var accepted = new List<FrozenTrial>();
        FrozenTrial[] splitting = [];
        foreach (var front in fronts)
        {
            // Core uses strict <; even an exactly fitting last front enters niching.
            if (accepted.Count + front.Length < config.PopulationSize)
                accepted.AddRange(front);
            else
            {
                splitting = front;
                break;
            }
        }

        var normalizedSet = accepted.Concat(splitting).ToArray();
        var (ideal, scales, method) = Normalize(normalizedSet);
        var refs = References(config.Divisions);
        // Using the same integer rays as the core also preserves its floating point tie behavior.
        var coreRefs = refs.Select(r => r.Select(v => Math.Round(v * config.Divisions)).ToArray()).ToArray();
        var selectedIds = selected.Select(t => t.Number).ToHashSet();
        var parentIds = parents.Select(t => t.Number).ToHashSet();
        var ranks = fronts.SelectMany((f, rank) => f.Select(t => (t.Number, rank)))
            .ToDictionary(t => t.Number, t => t.rank);
        var points = combined.Select(t =>
        {
            var values = t.Values!.ToArray();
            var normalized = values.Select((v, j) => (v - ideal[j]) * scales[j]).ToArray();
            var association = Associate(normalized, coreRefs);
            return new Point(t.Number, values, normalized, ranks[t.Number],
                association.Index, association.Distance, parentIds.Contains(t.Number) ? "parent" : "offspring",
                selectedIds.Contains(t.Number));
        }).ToDictionary(p => p.Id);

        var occupancy = new int[refs.Length];
        foreach (var trial in accepted)
            occupancy[points[trial.Number].ReferenceIndex]++;
        var acceptedIds = accepted.Select(t => t.Number).ToHashSet();
        var nicheSteps = selected.Where(t => !acceptedIds.Contains(t.Number)).Select(t =>
        {
            var point = points[t.Number];
            var before = occupancy[point.ReferenceIndex]++;
            return new NicheStep(t.Number, point.ReferenceIndex, before, point.Distance,
                before == 0 ? "Empty niche: choose the closest candidate to this direction."
                    : "Occupied niche: choose a remaining candidate using the sampler's seeded shuffle.");
        }).ToArray();
        var survivors = selected.Select(t => points[t.Number]).ToArray();
        var paretoIds = Rank(selected)[0].Select(t => t.Number).ToHashSet();
        var paretoFront = survivors.Where(p => paretoIds.Contains(p.Id)).ToArray();
        return new RunSnapshot(id, generation, (generation + 1) * config.PopulationSize, config,
            refs, parents.Select(t => points[t.Number]).ToArray(),
            offspring.Select(t => points[t.Number]).ToArray(), combined.Select(t => points[t.Number]).ToArray(),
            survivors, paretoFront,
            new Metrics(survivors.Average(p => Math.Abs(Math.Sqrt(p.Values.Sum(v => v * v)) - 1.0)),
                survivors.Select(p => p.ReferenceIndex).Distinct().Count(), refs.Length, paretoFront.Length),
            new Selection(fronts.Select(f => f.Select(t => t.Number).ToArray()).ToArray(),
                accepted.Select(t => t.Number).ToArray(), splitting.Select(t => t.Number).ToArray(),
                selected.Select(t => t.Number).ToArray(), ideal,
                scales.Select(v => v == 0 ? 0 : 1.0 / v).ToArray(),
                $"{method}. Geometry reconstructed from accepted fronts + splitting front; survivor IDs and niching order captured from OptunaSharp's actual selection. Later fronts use the same transform for display only.",
                nicheSteps));
    }

    internal static FrozenTrial[][] Rank(IReadOnlyList<FrozenTrial> population)
    {
        var remaining = population.ToList();
        var fronts = new List<FrozenTrial[]>();
        while (remaining.Count > 0)
        {
            var front = remaining.Where(a => !remaining.Any(b => Dominates(b.Values!, a.Values!))).ToArray();
            fronts.Add(front);
            var ids = front.Select(t => t.Number).ToHashSet();
            remaining.RemoveAll(t => ids.Contains(t.Number));
        }
        return fronts.ToArray();
    }

    private static bool Dominates(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var strict = false;
        for (var j = 0; j < a.Count; j++)
        {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) strict = true;
        }
        return strict;
    }

    private static (double[] Ideal, double[] Scales, string Method) Normalize(FrozenTrial[] population)
    {
        var ideal = Enumerable.Range(0, 3).Select(j => population.Min(t => t.Values![j])).ToArray();
        var shifted = population.Select(t => t.Values!.Select((v, j) => v - ideal[j]).ToArray()).ToArray();
        var extremes = new double[3, 3];
        for (var d = 0; d < 3; d++)
        {
            var best = 0;
            var bestAsf = double.PositiveInfinity;
            for (var i = 0; i < shifted.Length; i++)
            {
                var asf = Enumerable.Range(0, 3).Max(j => shifted[i][j] * (j == d ? 1.0 : 1e6));
                if (asf < bestAsf) { bestAsf = asf; best = i; }
            }
            for (var j = 0; j < 3; j++) extremes[d, j] = shifted[best][j];
        }
        using var extremeArray = np.array(extremes);
        using var rankArray = np.linalg.matrix_rank(extremeArray);
        var rank = (int)Convert.ToInt64(rankArray.GetValue(0));
        double[] scales;
        string method;
        if (rank == 3)
        {
            using var ones = np.array(new[] { 1.0, 1.0, 1.0 });
            using var solution = np.linalg.solve(extremeArray, ones);
            scales = solution.ToArray<double>();
            method = "ASF extreme-point hyperplane intercept normalization";
        }
        else
        {
            scales = Enumerable.Range(0, 3).Select(j => shifted.Max(p => p[j]))
                .Select(v => 1.0 / (v == 0 ? 1.0 : v)).ToArray();
            method = "Per-axis maximum fallback (degenerate ASF extreme points)";
        }
        for (var j = 0; j < 3; j++)
            if (!double.IsFinite(scales[j])) scales[j] = 1;
        return (ideal, scales, method);
    }

    internal static (int Index, double Distance, double[] Projection) Associate(double[] point, double[][] refs)
    {
        var best = 0;
        var bestDistance = double.PositiveInfinity;
        double[] bestProjection = [];
        for (var q = 0; q < refs.Length; q++)
        {
            var norm = Math.Sqrt(refs[q].Sum(v => v * v));
            var inv = 1.0 / (norm * norm);
            var projection = new double[3];
            var sum = 0.0;
            for (var k = 0; k < 3; k++)
            {
                for (var i = 0; i < 3; i++)
                    projection[k] += point[i] * refs[q][i] * inv * refs[q][k];
                var diff = point[k] - projection[k];
                sum += diff * diff;
            }
            var distance = Math.Sqrt(sum);
            if (distance < bestDistance)
            {
                best = q;
                bestDistance = distance;
                bestProjection = projection;
            }
        }
        return (best, bestDistance, bestProjection);
    }
}

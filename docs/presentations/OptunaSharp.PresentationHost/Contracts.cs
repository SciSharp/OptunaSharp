namespace OptunaSharp.PresentationHost;

public sealed record RunConfig(
    int PopulationSize = 36,
    int Divisions = 6,
    int Seed = 42,
    double MutationProbability = 0.15,
    string Problem = "dtlz2")
{
    public void Validate()
    {
        if (PopulationSize is < 12 or > 96)
            throw new ArgumentException("Population size must be between 12 and 96.");
        if (Divisions is < 1 or > 16)
            throw new ArgumentException("Reference divisions must be between 1 and 16.");
        if (Seed < 0)
            throw new ArgumentException("Seed must be a nonnegative 32-bit integer.");
        if (!double.IsFinite(MutationProbability) || MutationProbability is < 0 or > 1)
            throw new ArgumentException("Mutation probability must be finite and between 0 and 1.");
        if (!string.Equals(Problem, "dtlz2", StringComparison.Ordinal))
            throw new ArgumentException("The supported problem is dtlz2 (three objectives, twelve variables).");
    }
}

public sealed record Point(int Id, double[] Values, double[] Normalized, int Rank,
    int ReferenceIndex, double Distance, string Origin, bool Selected);

public sealed record NicheStep(int TrialId, int ReferenceIndex, int OccupancyBefore,
    double Distance, string Reason);

public sealed record Selection(int[][] Fronts, int[] AcceptedIds, int[] SplittingFrontIds,
    int[] SelectedIds, double[] IdealPoint, double[] Intercepts, string Normalization,
    NicheStep[] NicheSteps);

public sealed record Metrics(double MeanDistanceToFront, int OccupiedDirections,
    int ReferenceDirectionCount, int NonDominatedCount);

public sealed record RunSnapshot(string RunId, int Generation, int Evaluations, RunConfig Config,
    double[][] ReferenceDirections, Point[] Parents, Point[] Offspring, Point[] Candidates,
    Point[] Survivors, Point[] ParetoFront, Metrics Metrics, Selection Selection);

public sealed record ReferenceLabRequest(int Divisions = 6, double[]? Point = null);
public sealed record ReferenceLabResult(double[][] ReferenceDirections, double[] Point,
    int ReferenceIndex, double Distance, double[] Projection);

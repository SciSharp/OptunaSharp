using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.PresentationHost;

/// <summary>Dependency-free executable checks for algorithm integration, not a second optimizer.</summary>
internal static class SelfTest
{
    public static void Run()
    {
        var config = new RunConfig();
        var run = new Nsga3Run(config);
        var duplicate = new Nsga3Run(config);
        var baselineSampler = new NSGAIIISampler(populationSize: config.PopulationSize,
            mutation: new PolynomialMutation(20.0), mutationProb: config.MutationProbability,
            crossover: new SBXCrossover(eta: 20.0), crossoverProb: 0.9,
            seed: config.Seed, dividingParameter: config.Divisions);
        var baseline = Optuna.CreateStudy(sampler: baselineSampler,
            directions: [StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize]);

        // Run the baseline as uninterrupted, ordinary Ask/Tell. It must generate the same objective
        // values even though the host pauses at each generation to create presentation snapshots.
        var expected = new List<double[]>();
        for (var i = 0; i < config.PopulationSize * 5; i++)
        {
            var trial = baseline.Ask();
            var x = Enumerable.Range(0, 12).Select(j => trial.SuggestFloat($"x{j:00}", 0, 1)).ToArray();
            var values = Nsga3Run.EvaluateDtlz2(x);
            expected.Add(values);
            baseline.Tell(trial, values);
        }
        // Let the ordinary sampler reach the same pending first trial at the last boundary.
        // Its cached parent populations now expose all five real survivor sets without reselecting.
        baseline.Ask().SuggestFloat("x00", 0, 1);

        for (var generation = 0; generation < 5; generation++)
        {
            if (generation > 0) { run.Step(); duplicate.Step(); }
            var snapshot = run.Snapshot;
            ValidateSnapshot(snapshot);
            Check(run.SelectionCalls == generation + 1, "Only one real selection per generation boundary.");
            Check(ReferenceEquals(snapshot, run.Snapshot), "Read does not run or mutate the optimizer.");
            Check(snapshot.Selection.SelectedIds.SequenceEqual(duplicate.Snapshot.Selection.SelectedIds),
                "Same seed reproduces survivor ordering.");
            Check(snapshot.Selection.SelectedIds.Order().SequenceEqual(
                    baselineSampler.GetParentPopulation(baseline, generation + 1).Select(t => t.Number).Order()),
                "Survivor IDs match uninterrupted ordinary NSGAIIISampler exactly.");
            foreach (var point in snapshot.Offspring)
            {
                Check(point.Values.SequenceEqual(expected[point.Id]),
                    $"Trial {point.Id} matches uninterrupted NSGAIIISampler Ask/Tell exactly.");
                Check(point.Values.SequenceEqual(duplicate.Snapshot.Offspring.Single(p => p.Id == point.Id).Values),
                    "Same seed reproduces objective values.");
            }
        }

        var initialDistance = new Nsga3Run(config).Snapshot.Metrics.MeanDistanceToFront;
        while (run.Snapshot.Generation < 35)
        {
            run.Step();
            ValidateSnapshot(run.Snapshot);
        }
        Check(run.Snapshot.Metrics.MeanDistanceToFront < initialDistance * 0.65,
            "Real genetic evolution improves the default DTLZ2 population by generation 35.");

        var lab = SelectionGeometry.ReferenceLab(new ReferenceLabRequest(6, [1, 0, 0]));
        Check(lab.ReferenceDirections.Length == 28 && lab.ReferenceIndex == 0,
            "Three objectives and six divisions give 28 reference directions in core order.");
        Check(lab.Distance < 1e-12 && lab.Projection.SequenceEqual(new double[] { 1, 0, 0 }),
            "Reference-lab association projects an axial point onto its ray.");
        CheckThrows(() => new RunConfig(PopulationSize: 1).Validate(), "Reject invalid population size.");
        CheckThrows(() => new RunConfig(MutationProbability: double.NaN).Validate(), "Reject nonfinite probability.");
        CheckThrows(() => SelectionGeometry.ReferenceLab(new ReferenceLabRequest(Point: [double.PositiveInfinity, 0, 0])),
            "Reject nonfinite coordinates.");
        Console.WriteLine($"PASS: ordinary sampler parity (180 trials), deterministic replay, 36 generations of survivor/rank/niche invariants, input validation and reference geometry. Mean radial gap {initialDistance:F4} -> {run.Snapshot.Metrics.MeanDistanceToFront:F4}.");
    }

    private static void ValidateSnapshot(RunSnapshot snapshot)
    {
        var n = snapshot.Config.PopulationSize;
        Check(snapshot.Evaluations == (snapshot.Generation + 1) * n, "Count completed evaluations only.");
        Check(snapshot.Offspring.Length == n && snapshot.Survivors.Length == n, "Exactly N offspring and N survivors.");
        Check(snapshot.Candidates.Length == (snapshot.Generation == 0 ? n : 2 * n), "Correct parent-offspring union.");
        Check(snapshot.Candidates.Select(p => p.Id).Distinct().Count() == snapshot.Candidates.Length, "Candidates have unique IDs.");
        Check(snapshot.Selection.Fronts.SelectMany(f => f).Order().SequenceEqual(snapshot.Candidates.Select(p => p.Id).Order()),
            "Non-domination fronts partition the candidates.");
        Check(snapshot.Survivors.Select(p => p.Id).SequenceEqual(snapshot.Selection.SelectedIds), "Selected IDs are actual survivor order.");
        Check(snapshot.Selection.AcceptedIds.All(id => snapshot.Selection.SelectedIds.Contains(id)), "Complete earlier fronts all survive.");
        Check(snapshot.Candidates.All(p => p.Values.Concat(p.Normalized).Append(p.Distance).All(double.IsFinite)), "All geometry is finite.");
        Check(snapshot.Candidates.All(p => Math.Sqrt(p.Values.Sum(v => v * v)) >= 1.0 - 1e-12), "DTLZ2 objectives are on or outside the unit sphere.");

        var points = snapshot.Candidates.ToDictionary(p => p.Id);
        var occupancy = new int[snapshot.ReferenceDirections.Length];
        foreach (var id in snapshot.Selection.AcceptedIds) occupancy[points[id].ReferenceIndex]++;
        var remaining = snapshot.Selection.SplittingFrontIds.Select(id => points[id]).ToList();
        foreach (var step in snapshot.Selection.NicheSteps)
        {
            Check(remaining.Any(p => p.Id == step.TrialId), "Every niche choice belongs to the remaining splitting front.");
            var point = points[step.TrialId];
            Check(step.ReferenceIndex == point.ReferenceIndex, "Niche-step association agrees with geometry.");
            Check(step.OccupancyBefore == occupancy[step.ReferenceIndex], "Niche occupancy reconstructs actual selection order.");
            var least = remaining.Min(p => occupancy[p.ReferenceIndex]);
            Check(occupancy[step.ReferenceIndex] == least, "Actual sampler selects a least occupied available niche.");
            if (step.OccupancyBefore == 0)
                Check(point.Distance <= remaining.Where(p => p.ReferenceIndex == step.ReferenceIndex).Min(p => p.Distance) + 1e-12,
                    "The first occupant is nearest to its reference ray.");
            occupancy[step.ReferenceIndex]++;
            remaining.RemoveAll(p => p.Id == point.Id);
        }
        Check(snapshot.Selection.NicheSteps.Length + snapshot.Selection.AcceptedIds.Length == n,
            "Whole fronts plus niche choices fill the population.");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException("SELF-TEST FAILED: " + message);
    }

    private static void CheckThrows(Action action, string message)
    {
        try { action(); }
        catch (ArgumentException) { return; }
        throw new InvalidOperationException("SELF-TEST FAILED: " + message);
    }
}

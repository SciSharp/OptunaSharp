#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Solver configuration — tune a simulated-annealing TSP solver (start temperature, cooling schedule, move
// operator) across a benchmark set of 24 instances, with WilcoxonPruner abandoning a configuration as soon as
// a paired signed-rank test says it is significantly worse than the best one on the instances seen so far.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html
//         (the pruner, the "report one value per problem instance, step = instance id" protocol, and the
//         advice to return the running mean instead of raising TrialPruned)
//
// Real objective: each trial runs the annealer, with the trial's settings and a fixed move budget, on every
// instance (60 random Euclidean cities each, generated from a fixed seed) and scores the tour length relative
// to a nearest-neighbour + 2-opt reference tour — so 1.00 means "as good as the classical heuristic" and
// lower is better. This is algorithm configuration in the irace/SMAC sense: the instances are evaluated in a
// per-trial shuffled order, and the pruner compares each configuration against the incumbent on exactly the
// instances both have solved. Pruned configurations still report their running mean (as the Optuna docs
// recommend), so TPE learns from them instead of seeing a hole.
//
//   Run:  dotnet run examples/0_AnnealingSolverConfig.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var instances = Enumerable.Range(0, 24).Select(i => Tsp.Random(cities: 60, seed: 1000 + i)).ToArray();
double[] reference = instances.Select(Tsp.ReferenceLength).ToArray();   // NN + 2-opt, computed once

var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 21, nStartupTrials: 10),
    pruner: new WilcoxonPruner(pThreshold: 0.1, nStartupSteps: 4),   // judge only after 4 instances
    direction: StudyDirection.Minimize,
    studyName: "sa-tsp-config");
study.Optimize(t => Objective(t, instances, reference), nTrials: 60);

int solved = study.Trials.Sum(t => t.IntermediateValues.Count);
int pruned = study.Trials.Count(t => t.UserAttrs.TryGetValue("pruned_after", out _));
Console.WriteLine($"configurations: {study.Trials.Count}, cut short by the Wilcoxon test: {pruned}");
Console.WriteLine($"instance solves: {solved} of {study.Trials.Count * instances.Length} for an exhaustive evaluation");
Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
    "best mean tour / reference: {0:F4} (trial #{1}, evaluated on all {2} instances: {3})",
    study.BestValue, study.BestTrial.Number, instances.Length, study.BestTrial.IntermediateValues.Count == instances.Length));
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key,-12} {kv.Value}");

/// <summary>Runs one solver configuration over the instance set, reporting one result per instance.</summary>
/// <param name="t">The trial proposing the annealing schedule and move operator.</param>
/// <param name="instances">The benchmark instances.</param>
/// <param name="reference">Each instance's reference tour length.</param>
/// <returns>The mean tour-length ratio over the instances evaluated (all of them unless pruned).</returns>
static double Objective(Trial t, Tsp[] instances, double[] reference)
{
    double t0 = t.SuggestFloat("t0", 1e-3, 1.0, log: true);             // start temperature (× mean edge)
    double coolLog = t.SuggestFloat("cooling_log10", -5.0, -1.5);        // cooling = 1 − 10^x per move
    string move = t.SuggestCategorical("move", new[] { "two_opt", "swap", "insert" });
    double cooling = 1 - Math.Pow(10, coolLog);

    // a per-trial shuffled order: the pruner pairs by STEP (= instance id), not by evaluation order.
    var order = Enumerable.Range(0, instances.Length).ToArray();
    new Random(t.Number).Shuffle(order);
    var results = new List<double>();
    foreach (int id in order)
    {
        double ratio = instances[id].Anneal(t0, cooling, move, budget: 20_000, seed: id) / reference[id];
        results.Add(ratio);
        t.Report(ratio, id);
        if (t.ShouldPrune())
        {
            t.SetUserAttr("pruned_after", results.Count);
            return results.Average();   // the docs' workaround: tell the sampler what was measured
        }
    }
    return results.Average();
}

/// <summary>A symmetric Euclidean travelling-salesman instance with a simulated-annealing solver.</summary>
sealed class Tsp
{
    private readonly double[,] _d;
    private readonly double _meanEdge;

    /// <summary>The number of cities.</summary>
    public int N { get; }

    /// <summary>Builds the distance matrix for a set of points.</summary>
    /// <param name="xs">X coordinates.</param>
    /// <param name="ys">Y coordinates.</param>
    private Tsp(double[] xs, double[] ys)
    {
        N = xs.Length;
        _d = new double[N, N];
        double sum = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                _d[i, j] = Math.Sqrt((xs[i] - xs[j]) * (xs[i] - xs[j]) + (ys[i] - ys[j]) * (ys[i] - ys[j]));
                sum += _d[i, j];
            }
        _meanEdge = sum / (N * (N - 1));
    }

    /// <summary>Generates uniformly random cities in the unit square.</summary>
    /// <param name="cities">City count.</param>
    /// <param name="seed">Instance seed.</param>
    /// <returns>The instance.</returns>
    public static Tsp Random(int cities, int seed)
    {
        var rng = new Random(seed);   // instance generation only — samplers never use System.Random
        var xs = new double[cities];
        var ys = new double[cities];
        for (int i = 0; i < cities; i++)
            (xs[i], ys[i]) = (rng.NextDouble(), rng.NextDouble());
        return new Tsp(xs, ys);
    }

    /// <summary>Nearest-neighbour construction followed by 2-opt to a local optimum: the reference heuristic.</summary>
    /// <param name="tsp">The instance.</param>
    /// <returns>The reference tour's length.</returns>
    public static double ReferenceLength(Tsp tsp)
    {
        int n = tsp.N;
        var tour = new int[n];
        var used = new bool[n];
        used[0] = true;
        for (int k = 1; k < n; k++)
        {
            int prev = tour[k - 1], next = -1;
            for (int c = 0; c < n; c++)
                if (!used[c] && (next < 0 || tsp._d[prev, c] < tsp._d[prev, next]))
                    next = c;
            tour[k] = next;
            used[next] = true;
        }
        bool improved = true;
        while (improved)
        {
            improved = false;
            for (int i = 0; i < n - 1; i++)
                for (int j = i + 2; j < n; j++)
                    if (tsp.TwoOptDelta(tour, i, j) < -1e-12)
                    {
                        Array.Reverse(tour, i + 1, j - i);
                        improved = true;
                    }
        }
        return tsp.Length(tour);
    }

    /// <summary>Simulated annealing from a random tour with a geometric cooling schedule.</summary>
    /// <param name="t0">Start temperature as a multiple of the mean edge length.</param>
    /// <param name="cooling">Per-move temperature multiplier in (0, 1).</param>
    /// <param name="move">"two_opt" (segment reversal), "swap" (exchange two cities) or "insert" (move one city).</param>
    /// <param name="budget">Number of proposed moves.</param>
    /// <param name="seed">The annealer's own random seed (fixed per instance, so a configuration's score is reproducible).</param>
    /// <returns>The best tour length found.</returns>
    public double Anneal(double t0, double cooling, string move, int budget, int seed)
    {
        var rng = new Random(seed);
        var tour = Enumerable.Range(0, N).ToArray();
        rng.Shuffle(tour);
        double len = Length(tour), best = len, temp = t0 * _meanEdge;
        var trial = new int[N];
        for (int it = 0; it < budget; it++, temp *= cooling)
        {
            int i = rng.Next(N - 1), j = rng.Next(i + 1, N);
            double delta;
            if (move == "two_opt")
            {
                if (j == i + 1) continue;
                delta = TwoOptDelta(tour, i, j);
                if (delta < 0 || rng.NextDouble() < Math.Exp(-delta / temp))
                {
                    Array.Reverse(tour, i + 1, j - i);
                    len += delta;
                }
            }
            else
            {
                // swap / insert: build the candidate and measure it (O(n), simple and obviously correct).
                Array.Copy(tour, trial, N);
                if (move == "swap")
                    (trial[i], trial[j]) = (trial[j], trial[i]);
                else
                {
                    int city = trial[i];
                    Array.Copy(trial, i + 1, trial, i, j - i);
                    trial[j] = city;
                }
                delta = Length(trial) - len;
                if (delta < 0 || rng.NextDouble() < Math.Exp(-delta / temp))
                {
                    Array.Copy(trial, tour, N);
                    len += delta;
                }
            }
            best = Math.Min(best, len);
        }
        return best;
    }

    /// <summary>Length change of reversing the tour segment between positions i+1 and j.</summary>
    /// <param name="tour">The tour.</param>
    /// <param name="i">Position before the segment.</param>
    /// <param name="j">Last position of the segment (j &gt; i + 1).</param>
    /// <returns>New length minus old length.</returns>
    private double TwoOptDelta(int[] tour, int i, int j)
    {
        int a = tour[i], b = tour[i + 1], c = tour[j], d = tour[(j + 1) % N];
        return _d[a, c] + _d[b, d] - _d[a, b] - _d[c, d];
    }

    /// <summary>The closed tour's length.</summary>
    /// <param name="tour">A permutation of the cities.</param>
    /// <returns>Sum of consecutive distances, including the closing edge.</returns>
    private double Length(int[] tour)
    {
        double s = 0;
        for (int k = 0; k < N; k++)
            s += _d[tour[k], tour[(k + 1) % N]];
        return s;
    }
}

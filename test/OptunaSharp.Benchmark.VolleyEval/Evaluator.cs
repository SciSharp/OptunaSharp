using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The normalized score of one (challenge, contender) pair over a seed set, plus the two first-class
///     non-numeric outcomes the profile matrix records rather than hides: <see cref="Inert" /> (the
///     sampler's trajectory ≡ its independent random fallback) and <see cref="Dnf" /> (the sampler could
///     not run the challenge at all — e.g. a covariance sampler on a multi-objective study).
/// </summary>
public sealed record CellResult(
    string Challenge,
    string Contender,
    int EffectiveBudget,
    IReadOnlyList<double> SeedScores,
    double Median,
    double Q1,
    double Q3,
    bool Inert,
    bool Dnf,
    string? Error,
    double WallSeconds)
{
    /// <summary>The value used for ranking/rendering: NaN when the pair did not finish.</summary>
    public double Value => Dnf ? double.NaN : Median;
}

/// <summary>
///     Runs the VolleyEval rounds and aggregates them. A <b>round</b> is one (challenge, contender, seed)
///     study; a <see cref="CellResult" /> aggregates the rounds over seeds (median + IQR, per
///     Best-Practice §5: report seed distributions, never bare means). This is the whole v0
///     apparatus — construct study, optimize to budget, normalize via the challenge, and detect
///     inert/DNF.
/// </summary>
public sealed class Evaluator
{
    private readonly int[] _seeds;
    private readonly double? _maxSecondsPerRun;

    // Cache of RandomSampler(seed) parameter trajectories, keyed by (challenge, seed, nTrials), so the
    // inert comparison does not re-run random for every contender.
    private readonly Dictionary<(string, int, int), List<string>> _randomTrajectoryCache = new();

    /// <param name="seeds">The search seeds. Best-Practice §5 asks for ≥ 5 for a stable median + IQR.</param>
    /// <param name="maxSecondsPerRun">Optional wall-clock safety cap per study (passed to <c>Study.Optimize(timeout:)</c>). <c>null</c> = rely on trial budgets alone (fully reproducible).</param>
    public Evaluator(int[] seeds, double? maxSecondsPerRun = null)
    {
        if (seeds.Length == 0) throw new ArgumentException("At least one seed is required.", nameof(seeds));
        _seeds = seeds;
        _maxSecondsPerRun = maxSecondsPerRun;
    }

    /// <summary>Evaluate every (challenge × contender) pair into a <see cref="ProfileMatrix" />.</summary>
    public ProfileMatrix Run(
        IReadOnlyList<IChallenge> challenges,
        IReadOnlyList<Contender> contenders,
        Action<CellResult>? onCell = null)
    {
        var cells = new List<CellResult>();
        foreach (var challenge in challenges)
            foreach (var contender in contenders)
            {
                var cell = EvaluateCell(challenge, contender);
                cells.Add(cell);
                onCell?.Invoke(cell);
            }
        return new ProfileMatrix(challenges, contenders, cells);
    }

    /// <summary>Evaluate a single (challenge, contender) pair over all seeds.</summary>
    public CellResult EvaluateCell(IChallenge challenge, Contender contender)
    {
        int nTrials = contender.EffectiveBudget(challenge.Budget);
        var scores = new List<double>();
        string? error = null;
        bool inert = challenge.DetectInert && contender.Name != "random";
        var sw = Stopwatch.StartNew();

        foreach (int seed in _seeds)
        {
            Study study;
            try
            {
                study = RunStudy(challenge, contender.BuildSampler(seed), nTrials);
            }
            catch (Exception ex)
            {
                error = ex.GetType().Name;
                inert = false;
                continue; // this seed did-not-finish; other seeds may still succeed
            }

            scores.Add(challenge.Score(study));

            if (inert)
            {
                var trajectory = ParameterTrajectory(study);
                var randomTrajectory = RandomTrajectory(challenge, seed, nTrials);
                if (!TrajectoriesEqual(trajectory, randomTrajectory))
                    inert = false;
            }
        }
        sw.Stop();

        bool dnf = scores.Count == 0;
        double median = dnf ? double.NaN : Percentile(scores, 50);
        double q1 = dnf ? double.NaN : Percentile(scores, 25);
        double q3 = dnf ? double.NaN : Percentile(scores, 75);

        return new CellResult(
            challenge.Name, contender.Name, nTrials,
            scores, median, q1, q3, inert, dnf, error, sw.Elapsed.TotalSeconds);
    }

    private Study RunStudy(IChallenge challenge, BaseSampler sampler, int nTrials)
    {
        if (challenge.NumObjectives == 1)
        {
            var study = Optuna.CreateStudy(sampler: sampler, direction: challenge.Directions[0]);
            study.Optimize(t => challenge.Evaluate(t)[0], nTrials: nTrials, timeout: _maxSecondsPerRun);
            return study;
        }
        else
        {
            var study = Optuna.CreateStudy(sampler: sampler, directions: challenge.Directions);
            study.Optimize(t => challenge.Evaluate(t), nTrials: nTrials, timeout: _maxSecondsPerRun);
            return study;
        }
    }

    private List<string> RandomTrajectory(IChallenge challenge, int seed, int nTrials)
    {
        var key = (challenge.Name, seed, nTrials);
        if (_randomTrajectoryCache.TryGetValue(key, out var cached))
            return cached;
        var study = RunStudy(challenge, new RandomSampler(seed), nTrials);
        var traj = ParameterTrajectory(study);
        _randomTrajectoryCache[key] = traj;
        return traj;
    }

    // A canonical, order-preserving rendering of each trial's suggested parameters — the "trajectory"
    // whose equality to the RandomSampler fallback is the exact definition of "inert" (≡ random).
    private static List<string> ParameterTrajectory(Study study)
        => study.Trials
            .OrderBy(t => t.Number)
            .Select(t => string.Join("|", t.Params.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                .Select(kv => $"{kv.Key}={kv.Value}")))
            .ToList();

    private static bool TrajectoriesEqual(List<string> a, List<string> b)
        => a.Count == b.Count && a.Zip(b, (x, y) => x == y).All(eq => eq);

    /// <summary>Linear-interpolation percentile (numpy "type-7"), used for the median and IQR.</summary>
    public static double Percentile(IReadOnlyList<double> values, double p)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        if (sorted.Length == 1) return sorted[0];
        double rank = p / 100.0 * (sorted.Length - 1);
        int lo = (int)Math.Floor(rank);
        int hi = (int)Math.Ceiling(rank);
        double frac = rank - lo;
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }
}

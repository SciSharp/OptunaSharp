using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The result of tuning one (challenge, contender) profile: the held-out score of the default
///     (paper) configuration against the held-out score of the tuned configuration, plus the winning
///     recipe. The held-out figure is the honest one — a config chosen on the search seeds is re-scored
///     on <b>unseen</b> seeds, so a negative <see cref="Delta" /> is a real "tuning overfit / did not
///     transfer" finding, not a bug.
/// </summary>
public sealed record TunedProfile(
    string Challenge,
    string Contender,
    bool Tunable,
    bool Dnf,
    double DefaultScore,   // median over held-out seeds, default config, full budget
    double TunedScore,     // median over held-out seeds, tuned config, full budget
    double SearchScore,    // best median over search seeds during tuning (the outer optimum)
    IReadOnlyDictionary<string, object?> Recipe)
{
    /// <summary>Tuned minus default on the held-out seeds (the value that matters).</summary>
    public double Delta => TunedScore - DefaultScore;

    /// <summary>True when tuning produced a genuine held-out gain.</summary>
    public bool Improved => Tunable && !Dnf && Delta > 1e-6;

    /// <summary>True when the search-seed optimum overstated the held-out result (the guard's whole point).</summary>
    public bool Overfit => Tunable && !Dnf && SearchScore - TunedScore > 0.02;
}

/// <summary>
///     The nested-hyperopt protocol (PLAN.md §3), "optimize the optimizer": for each profile an
///     <b>outer</b> TPE study searches the contender's declared knob space, its objective runs the
///     challenge at budget over the <b>search seeds</b>, and the winning recipe is re-scored on
///     <b>held-out seeds</b> to guard against a single-seed fluke. Both levels are OptunaSharp studies
///     — <see cref="Optuna.CreateStudy" /> wrapping <see cref="Optuna.CreateStudy" />.
/// </summary>
public sealed class HyperOptimizer
{
    private readonly int[] _searchSeeds;
    private readonly int[] _heldoutSeeds;
    private readonly int _nTuneTrials;
    private readonly int? _tuneBudgetCap;
    private readonly double? _maxSecondsPerRun;
    private readonly int _outerSeed;

    /// <param name="searchSeeds">Seeds the outer objective averages over while searching (the "train" seeds).</param>
    /// <param name="heldoutSeeds">Disjoint seeds used only for the final unbiased comparison.</param>
    /// <param name="nTuneTrials">Outer TPE budget (configurations tried per profile).</param>
    /// <param name="tuneBudgetCap">
    ///     Caps the <i>inner</i> per-run budget <b>during tuning</b> for speed (validation always uses the
    ///     challenge's full budget). Tuning at a reduced budget is a deliberate, documented trade-off:
    ///     the held-out validation at full budget then reveals whether the recipe transfers.
    /// </param>
    /// <param name="maxSecondsPerRun">Optional wall-clock safety cap per inner study.</param>
    /// <param name="outerSeed">Seed for the outer TPE sampler (reproducible search).</param>
    public HyperOptimizer(
        int[] searchSeeds, int[] heldoutSeeds, int nTuneTrials,
        int? tuneBudgetCap = null, double? maxSecondsPerRun = null, int outerSeed = 0)
    {
        if (searchSeeds.Length == 0 || heldoutSeeds.Length == 0)
            throw new ArgumentException("Both search and held-out seed sets must be non-empty.");
        if (searchSeeds.Intersect(heldoutSeeds).Any())
            throw new ArgumentException("Search and held-out seeds must be disjoint (that is the whole guard).");
        _searchSeeds = searchSeeds;
        _heldoutSeeds = heldoutSeeds;
        _nTuneTrials = nTuneTrials;
        _tuneBudgetCap = tuneBudgetCap;
        _maxSecondsPerRun = maxSecondsPerRun;
        _outerSeed = outerSeed;
    }

    /// <summary>Tune every (challenge × contender) profile.</summary>
    public IReadOnlyList<TunedProfile> Run(
        IReadOnlyList<IChallenge> challenges,
        IReadOnlyList<Contender> contenders,
        Action<TunedProfile>? onProfile = null)
    {
        var results = new List<TunedProfile>();
        foreach (var challenge in challenges)
            foreach (var contender in contenders)
            {
                var profile = Tune(challenge, contender);
                results.Add(profile);
                onProfile?.Invoke(profile);
            }
        return results;
    }

    /// <summary>Tune a single profile: default held-out score, outer search, tuned held-out score, recipe.</summary>
    public TunedProfile Tune(IChallenge challenge, Contender contender)
    {
        int fullBudget = contender.EffectiveBudget(challenge.Budget);

        // 1) Default (paper) configuration, scored on the held-out seeds at full budget.
        double defaultScore = MedianHeldout(challenge, seed => contender.BuildSampler(seed), fullBudget);
        var emptyRecipe = new Dictionary<string, object?>();

        // A contender whose default cannot even run this challenge (e.g. CMA on multi-objective) is a DNF
        // profile — nothing to tune.
        if (double.IsNaN(defaultScore))
            return new TunedProfile(challenge.Name, contender.Name, contender.Tunable, Dnf: true,
                double.NaN, double.NaN, double.NaN, emptyRecipe);

        // A contender with no knobs (random) is reported as-is — tuned == default.
        if (!contender.Tunable)
            return new TunedProfile(challenge.Name, contender.Name, Tunable: false, Dnf: false,
                defaultScore, defaultScore, defaultScore, emptyRecipe);

        // 2) Outer TPE search over the knob space, objective = median search-seed score at the tune budget.
        int tuneBudget = contender.EffectiveBudget(
            _tuneBudgetCap is int cap ? Math.Min(challenge.Budget, cap) : challenge.Budget);

        var outer = Optuna.CreateStudy(sampler: new TPESampler(seed: _outerSeed), direction: StudyDirection.Maximize);
        outer.Optimize(trial =>
        {
            var scores = new List<double>();
            foreach (int seed in _searchSeeds)
            {
                challenge.Reset(seed);
                BaseSampler sampler;
                try { sampler = contender.TunedBuild!(trial, seed); }
                catch { return double.NegativeInfinity; } // an invalid knob combination
                try
                {
                    var study = Evaluator.RunChallengeStudy(challenge, sampler, tuneBudget, _maxSecondsPerRun);
                    scores.Add(challenge.Score(study));
                }
                catch { return double.NegativeInfinity; } // this configuration did not finish
            }
            return Median(scores);
        }, nTrials: _nTuneTrials);

        var recipe = outer.BestTrial.ParamsDict;
        double searchScore = outer.BestValue;

        // 3) Held-out validation of the winning recipe, at FULL budget, on unseen seeds.
        var fixedTrial = new FixedTrial(recipe);
        double tunedScore = MedianHeldout(challenge, seed => contender.TunedBuild!(fixedTrial, seed), fullBudget);

        return new TunedProfile(challenge.Name, contender.Name, Tunable: true, Dnf: false,
            defaultScore, tunedScore, searchScore, recipe);
    }

    private double MedianHeldout(IChallenge challenge, Func<int, BaseSampler> build, int budget)
    {
        var scores = new List<double>();
        foreach (int seed in _heldoutSeeds)
        {
            challenge.Reset(seed);
            try
            {
                var study = Evaluator.RunChallengeStudy(challenge, build(seed), budget, _maxSecondsPerRun);
                scores.Add(challenge.Score(study));
            }
            catch { /* this seed did not finish */ }
        }
        return scores.Count == 0 ? double.NaN : Median(scores);
    }

    private static double Median(List<double> xs)
    {
        if (xs.Count == 0) return double.NaN;
        var s = xs.OrderBy(v => v).ToArray();
        return s[s.Length / 2];
    }
}

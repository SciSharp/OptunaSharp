using System;
using System.Collections.Generic;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Tpe;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 14: TPE keeps finished trials' internal parameter values in a <see cref="TpeObservationCache" /> instead of
///     converting the whole history on every ask. Each test runs the same seeded study twice, once with the cache warm
///     (as sampling uses it) and once with it emptied before every suggestion (so every observation is converted from the
///     trial again, as before the cache), and asserts every trial is identical to the bit: dynamic ranges (one name, many
///     distributions), conditional parameters (finished trials lacking a name), pruned trials, constant liar with trials
///     still running, both TPE modes, two objectives and compact storage.
/// </summary>
[TestClass]
public class TpeObservationCacheTests
{
    /// <summary>Every trial's number, state, parameters (value bits) and values (bits), one line per trial.</summary>
    /// <param name="study">The finished study.</param>
    /// <returns>The study's trials as text.</returns>
    private static string Dump(Study study)
    {
        var sb = new StringBuilder();
        foreach (var t in study.Trials)
        {
            sb.Append(t.Number).Append(' ').Append(t.State).Append(' ');
            foreach (var (k, v) in t.Params)
                sb.Append(k).Append('=').Append(v is double d ? BitConverter.DoubleToInt64Bits(d).ToString("X16") : v?.ToString()).Append(';');
            if (t.Values is not null)
                foreach (double v in t.Values)
                    sb.Append(' ').Append(BitConverter.DoubleToInt64Bits(v).ToString("X16"));
            sb.Append('\n');
        }
        return sb.ToString();
    }

    /// <summary>
    ///     The objective with a dynamic range, a conditional parameter and pruning; <paramref name="beforeSuggest" /> runs
    ///     before each suggestion (the cold run empties the cache there).
    /// </summary>
    /// <param name="t">The trial.</param>
    /// <param name="beforeSuggest">Called before each suggestion.</param>
    /// <returns>The objective value.</returns>
    /// <exception cref="TrialPruned">Every seventh trial is pruned after reporting its value.</exception>
    private static double Objective(Trial t, Action beforeSuggest)
    {
        double span = 5.0 + t.Number % 3;
        beforeSuggest();
        double x = t.SuggestFloat("x", -span, span);
        beforeSuggest();
        string c = t.SuggestCategorical("c", new[] { "a", "b", "c" });
        double y = 0.0;
        if (c == "a")
        {
            beforeSuggest();
            y = t.SuggestFloat("y", 1e-3, 10.0, log: true);
        }
        beforeSuggest();
        long k = t.SuggestInt("k", 0, 20, step: 2);
        double value = x * x + y + k / 10.0 + (c == "b" ? 0.0 : 1.0);
        t.Report(value, 0);
        if (t.Number % 7 == 3)
            throw new TrialPruned();
        return value;
    }

    /// <summary>The study run warm and cold must be identical, for both TPE modes, with plain and compact storage.</summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly.</param>
    /// <param name="compact">Whether the storage keeps finished trials compact (plan 13, item S12).</param>
    [TestMethod]
    [DataRow(false, false)]
    [DataRow(true, false)]
    [DataRow(false, true)]
    [DataRow(true, true)]
    public void WarmCache_EqualsColdCache_DynamicConditionalPruned(bool multivariate, bool compact)
    {
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 21, multivariate: multivariate, nStartupTrials: 5);
            var study = Optuna.CreateStudy(storage: new InMemoryStorage(compactFinishedTrials: compact), sampler: sampler);
            Action before = cold ? () => sampler.ObservationCache.Reset() : () => { };
            study.Optimize(t => Objective(t, before), nTrials: 120);
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
    }

    /// <summary>Two objectives (MOTPE), warm and cold, must be identical.</summary>
    [TestMethod]
    public void WarmCache_EqualsColdCache_MultiObjective()
    {
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 5, nStartupTrials: 5);
            var study = Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
            study.Optimize(t =>
            {
                double a = 0, b = 0;
                for (int i = 0; i < 6; i++)
                {
                    if (cold)
                        sampler.ObservationCache.Reset();
                    double v = t.SuggestFloat("x" + i, -3, 3);
                    a += v * v;
                    b += (v - 1) * (v - 1);
                }
                return new[] { a, b };
            }, nTrials: 100);
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
    }

    /// <summary>
    ///     Constant liar with trials still running (ask three, then tell three): running trials must never be cached, and
    ///     warm and cold runs must be identical in both TPE modes.
    /// </summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly.</param>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void WarmCache_EqualsColdCache_ConstantLiarWithRunningTrials(bool multivariate)
    {
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 9, multivariate: multivariate, constantLiar: true, nStartupTrials: 4);
            var study = Optuna.CreateStudy(sampler: sampler);
            for (int round = 0; round < 25; round++)
            {
                var batch = new List<Trial>();
                for (int i = 0; i < 3; i++)
                {
                    var t = study.Ask();
                    if (cold)
                        sampler.ObservationCache.Reset();
                    t.SuggestFloat("x", -5, 5);
                    if (cold)
                        sampler.ObservationCache.Reset();
                    t.SuggestCategorical("c", new[] { "u", "v", "w" });
                    batch.Add(t);
                }
                foreach (var t in batch)
                {
                    var p = study.Storage.GetTrial(t.TrialId).Params;
                    study.Tell(t, (double)p["x"]! * (double)p["x"]! + ((string)p["c"]! == "u" ? 0.0 : 1.0));
                }
            }
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
    }

    /// <summary>
    ///     A range that changes on every trial makes a new column per distribution; the cache must stay within
    ///     <see cref="TpeObservationCache.MaxColumns" /> by starting over, and the study must equal the cold run.
    /// </summary>
    [TestMethod]
    public void EverChangingRange_StaysBounded_AndEqualsColdCache()
    {
        int maxSeen = 0;
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 4, nStartupTrials: 3);
            var study = Optuna.CreateStudy(sampler: sampler);
            study.Optimize(t =>
            {
                if (cold)
                    sampler.ObservationCache.Reset();
                double x = t.SuggestFloat("x", 0, 1 + t.Number);
                maxSeen = Math.Max(maxSeen, sampler.ObservationCache.ColumnCount);
                return x;
            }, nTrials: 400);
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
        maxSeen.Should().BeLessThanOrEqualTo(TpeObservationCache.MaxColumns);
    }

    /// <summary>
    ///     One sampler reused for a second study (trial numbers start over) gives the second study exactly what a new sampler
    ///     with the same seed state would: the cache must start over when the study changes.
    /// </summary>
    [TestMethod]
    public void SecondStudy_StartsTheCacheOver()
    {
        var reused = new TPESampler(seed: 8, nStartupTrials: 3);
        var first = Optuna.CreateStudy(sampler: reused);
        first.Optimize(t => Math.Pow(t.SuggestFloat("x", -2, 2) - 1, 2), nTrials: 40);
        var second = Optuna.CreateStudy(sampler: reused);
        second.Optimize(t => Math.Pow(t.SuggestFloat("x", -2, 2) + 1, 2), nTrials: 40);

        // The same sequence with the cache emptied before every suggestion: identical, so nothing carried over.
        var cold = new TPESampler(seed: 8, nStartupTrials: 3);
        var coldFirst = Optuna.CreateStudy(sampler: cold);
        coldFirst.Optimize(t => { cold.ObservationCache.Reset(); return Math.Pow(t.SuggestFloat("x", -2, 2) - 1, 2); }, nTrials: 40);
        var coldSecond = Optuna.CreateStudy(sampler: cold);
        coldSecond.Optimize(t => { cold.ObservationCache.Reset(); return Math.Pow(t.SuggestFloat("x", -2, 2) + 1, 2); }, nTrials: 40);

        Dump(first).Should().Be(Dump(coldFirst));
        Dump(second).Should().Be(Dump(coldSecond));
    }
}

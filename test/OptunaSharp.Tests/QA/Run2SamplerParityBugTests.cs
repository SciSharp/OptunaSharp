using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Sampler parity bugs found by the 2026-11-10 parity QA run (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Expected
///     values are what Optuna 5.0.0 produces on CPython 3.12 / NumPy 2.4.2. A test marked <c>[OpenBugs]</c> fails until
///     its divergence is fixed; the attribute is removed with the fix, and the test then guards against a regression.
/// </summary>
[TestClass]
public class Run2SamplerParityBugTests
{
    /// <summary>
    ///     Runs a three-trial NSGA-III study (population 2, two maximized objectives) whose first generation returns
    ///     <c>(+inf, +inf)</c> for both trials, so the second generation's parents come from a front whose normalized
    ///     objectives are all NaN.
    /// </summary>
    /// <param name="seed">The sampler seed; the same seed drives Optuna's run that produced the expected values.</param>
    /// <returns>The finished study; its third trial is the first child of the all-infinite generation.</returns>
    /// <remarks>
    ///     Both objectives are infinite for every member of the borderline front, so <c>_filter_inf</c> finds no finite
    ///     value in either column and returns NaN everywhere, and every distance to a reference line is NaN.
    /// </remarks>
    private static Study RunAllInfiniteFirstGeneration(int seed)
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Maximize, StudyDirection.Maximize },
            sampler: new NSGAIIISampler(populationSize: 2, seed: seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            double y = t.SuggestFloat("y", 0, 1);
            // Generation 0 (trials 0 and 1) is entirely infinite; later trials return their parameters.
            return t.Number < 2
                ? new[] { double.PositiveInfinity, double.PositiveInfinity }
                : new[] { x, y };
        }, nTrials: 3);
        return study;
    }

    /// <summary>
    ///     QA-56: NSGA-III's niche preservation sorts the borderline front's <c>(distance, index)</c> pairs with
    ///     Python's <c>list.sort(reverse=True)</c>, under which a pair holding a NaN distance is neither smaller nor larger
    ///     than another, so timsort keeps such pairs in their original order and <c>pop()</c> takes the last one first.
    ///     The port's comparer orders NaN as equal to NaN and then by index, which reverses them: the elite population
    ///     comes out as <c>[T0, T1]</c> where Optuna builds <c>[T1, T0]</c>, and the next child copies the other parent's
    ///     genes.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0, seed 0: the third trial is <c>x=0.4236547993389047, y=0.7151893663724195</c>; the port gives
    ///     <c>y=0.5448831829968969</c> (the other parent's value). Seed 3: Optuna <c>x=0.5507979025745755,
    ///     y=0.510827605197663</c>, the port <c>x=0.2909047389129443, y=0.7081478226181048</c>. Instrumenting Optuna's
    ///     <c>_preserve_niche_individuals</c> shows closest reference points <c>[0, 0]</c>, distances <c>[nan, nan]</c>
    ///     and the order <c>[1, 0]</c>. Any front whose members are all non-finite in one objective (an objective that
    ///     returns <c>inf</c> to signal a failed evaluation) reaches this path. Finite ties sort identically in both.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Nsga3_AllInfiniteBorderlineFront_KeepsPythonsNichingOrder()
    {
        var seed0 = RunAllInfiniteFirstGeneration(0).Trials[2];
        ((double)seed0.Params["x"]!).Should().Be(0.4236547993389047);
        ((double)seed0.Params["y"]!).Should().Be(0.7151893663724195);

        var seed3 = RunAllInfiniteFirstGeneration(3).Trials[2];
        ((double)seed3.Params["x"]!).Should().Be(0.5507979025745755);
        ((double)seed3.Params["y"]!).Should().Be(0.510827605197663);
    }

    /// <summary>
    ///     Runs 19 trials of a seeded genetic sampler through Optuna's documented batch pattern (ask five trials, evaluate
    ///     them, tell all five) on one thread, over a categorical and a log-scaled float parameter.
    /// </summary>
    /// <param name="sampler">A fresh NSGA-II or NSGA-III sampler (population 6, seed 311031104).</param>
    /// <returns>The finished study; trials 15 to 18 form the last batch, all running while each is sampled.</returns>
    /// <remarks>The objective is a fixed function of the two values, so both languages tell identical values.</remarks>
    private static Study RunBatchedAskTell(BaseSampler sampler)
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: sampler);
        var choices = new OptunaSharp.Distributions.CategoricalChoice[] { 1, "a" };
        int n = 0;
        while (n < 19)
        {
            // Ask a whole batch before telling any of it, as Optuna's "Batch Optimization" tutorial does.
            var batch = new System.Collections.Generic.List<(Trial Trial, double Value)>();
            for (int b = 0; b < 5 && n < 19; b++, n++)
            {
                var trial = study.Ask();
                var c = trial.SuggestCategorical("lr2", choices);
                double v = trial.SuggestFloat("v5", 1e-05, 1.0, log: true);
                double a = v * 0.37 - 1;
                batch.Add((trial, (c.Equals(choices[0]) ? 0 : 1) * 0.5 + a * a));
            }
            foreach (var (trial, value) in batch)
                study.Tell(trial, value);
        }
        return study;
    }

    /// <summary>
    ///     QA-57: the genetic samplers' running-trial redraw (plan 12, <c>HandedOutParams.Claim</c>) changes the seeded
    ///     trajectory of a single-threaded batched ask/tell, and it calls two children duplicates when mutation dropped a
    ///     parameter that each trial then samples on its own. Trial 18's child is <c>{lr2: 'a'}</c> (its <c>v5</c> was
    ///     mutated away, so <c>v5</c> comes from the random sampler); trial 15, still running, holds the same partial
    ///     child, so the port generates three more children and keeps the last one, whose <c>v5</c> is inherited from
    ///     trial 1. Optuna generates one child and samples <c>v5</c> independently.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0 (NSGA-II and NSGA-III alike): trial 18 is <c>lr2='a', v5=0.02621137972316322</c>; the port gives
    ///     <c>v5=0.00021556396515864647</c> (trial 1's value). Every random draw up to trial 18's mutation decisions is
    ///     identical (MT19937 states compared at every ask); the extra <c>_child_generation_strategy</c> calls are the
    ///     divergence. THREADING.md states that runs asking one trial at a time keep Optuna parity; this run asks on one
    ///     thread too, but keeps five trials running, which Optuna's ask-and-tell tutorial recommends for batch
    ///     evaluation. Even under real concurrency the two partial children are not one point.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Nsga_BatchedAskTell_PartialChildIsNotRedrawn()
    {
        foreach (var sampler in new BaseSampler[]
                 {
                     new NSGAIISampler(populationSize: 6, seed: 311031104),
                     new NSGAIIISampler(populationSize: 6, seed: 311031104),
                 })
        {
            var trial = RunBatchedAskTell(sampler).Trials[18];
            trial.Params["lr2"].Should().Be("a");
            ((double)trial.Params["v5"]!).Should().Be(0.02621137972316322, sampler.GetType().Name);
        }
    }
}

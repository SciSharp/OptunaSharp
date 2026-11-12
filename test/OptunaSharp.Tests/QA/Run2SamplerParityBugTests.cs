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
    ///     Optuna 5.0.0 (NSGA-II and NSGA-III alike): trial 18 is <c>lr2='a', v5=0.02621137972316322</c>; the port gave
    ///     <c>v5=0.00021556396515864647</c> (trial 1's value). Every random draw up to trial 18's mutation decisions is
    ///     identical (MT19937 states compared at every ask); the extra <c>_child_generation_strategy</c> calls were the
    ///     divergence. THREADING.md states that runs asking one trial at a time keep Optuna parity; this run asks on one
    ///     thread too, but keeps five trials running, which Optuna's ask-and-tell tutorial recommends for batch
    ///     evaluation. Even under real concurrency the two partial children are not one point.
    ///     <para>
    ///         Fixed 2026-11-12 (owner's decision): a child is redrawn only when a running trial asked on another thread
    ///         holds it (<c>HandedOutParams</c> records the claiming thread), so a batch asked on one thread draws exactly
    ///         what Optuna draws, while concurrent workers stay steered apart (<c>SamplerParallelReadTests</c>).
    ///     </para>
    /// </remarks>
    [TestMethod]
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

    /// <summary>
    ///     QA-60: CMA-ES with margin feeds <c>CMAwM.tell</c> the raw <c>x_for_tell</c> vectors the generation's trials
    ///     stored. When the search space loses a dimension before the first generation is told (here <c>y</c>'s range
    ///     changes on trial 2, so the intersection search space keeps only <c>x</c> and a new one-dimensional optimizer
    ///     starts at generation 0), that generation mixes two-element and one-element vectors. Optuna's
    ///     <c>np.array([s[0] for s in solutions])</c> refuses the ragged list and the trial fails with <c>ValueError:
    ///     setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.
    ///     …</c>; the port's <c>Cma.Tell</c> reads the first <c>n_dim</c> entries of each vector, so the old
    ///     two-dimensional points are told with their first coordinate standing for a different parameter, and the run
    ///     carries on.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0 / cmaes 0.13.1, seed 0, 12 trials (<c>x</c> int in [0, 10], <c>y</c> float in [0, 1], [0, 1.5]
    ///     on trial 2 only): population 3 raises at trial 4, population 4 at trial 5 (the inhomogeneous-shape error),
    ///     population 2 at trial 3 (<c>non-broadcastable output operand with shape (1,) doesn't match the broadcast
    ///     shape (2,)</c>, the uniform-length variant). The port finishes all 12 trials in each case. Optuna's crash is
    ///     arguably a defect of its own, but the port's silent misaligned update is no better; the owner decides whether
    ///     to raise as Optuna does or to guard the tell. Found by the dynamic-range program fuzz (11 of 68 CMA-ES
    ///     programs, all with margin), then by a log-stream comparison with a merely conditional parameter: any
    ///     parameter that only some trials suggest makes Optuna's CMA-ES-with-margin raise at the end of the first
    ///     generation, which makes this the common path, not an exotic one.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CmaWithMargin_SearchSpaceShrinksInFirstGeneration_RaisesLikeOptuna()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(withMargin: true, seed: 0, popsize: 3));
        System.Action run = () =>
        {
            for (int i = 0; i < 12; i++)
            {
                var trial = study.Ask();
                long x = trial.SuggestInt("x", 0, 10);
                // The range changes once, so `y` leaves the intersection search space from trial 2 on.
                double y = trial.SuggestFloat("y", 0.0, i == 2 ? 1.5 : 1.0);
                study.Tell(trial, x + y);
            }
        };
        run.Should().Throw<NumSharp.ValueError>()
            .WithMessage("setting an array element with a sequence.*inhomogeneous shape*");
        study.Trials.Should().HaveCount(5, "Optuna fails on the fifth ask (trial 4)");

        // The same happens with a conditional parameter, no range change needed: `y` exists on even trials only, so the
        // intersection search space drops it once trial 1 completes. Optuna raises at trial 4 for seeds 0, 1 and 2.
        var conditional = Optuna.CreateStudy(sampler: new CmaEsSampler(withMargin: true, seed: 0, popsize: 3));
        System.Action runConditional = () =>
        {
            for (int i = 0; i < 20; i++)
            {
                var trial = conditional.Ask();
                long x = trial.SuggestInt("x", 0, 10);
                double y = i % 2 == 0 ? trial.SuggestFloat("y", 0.0, 1.0) : 0.0;
                conditional.Tell(trial, x + y);
            }
        };
        runConditional.Should().Throw<NumSharp.ValueError>()
            .WithMessage("setting an array element with a sequence.*inhomogeneous shape*");
        conditional.Trials.Should().HaveCount(5);
    }

    /// <summary>
    ///     QA-68: a trial numbered at or above the grid size picks its grid with
    ///     <c>rng.choice(list(set(range(n)) - set(visited) - set(running)))</c>, so the candidates come in CPython's
    ///     set-iteration order: the difference builds a fresh set whose small table makes values of 8 and more wrap onto
    ///     the slots of smaller ones, so <c>list(set(range(16)) - {0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 15})</c> is
    ///     <c>[8, 11, 4]</c> in CPython 3.12. The port lists the candidates in ascending order, so the same random index
    ///     picks another grid.
    /// </summary>
    /// <remarks>
    ///     The port marks this <c>[Misaligned]</c> in <c>GridSampler.GetUnvisitedGridIds</c>, reasoning that only real
    ///     concurrency reaches the <c>choice</c> branch. A single-threaded <c>optimize</c> reaches it whenever the
    ///     study's numbering is shifted: earlier trials from <c>add_trial</c> / <c>enqueue_trial</c>, another sampler,
    ///     or a resumed study. Optuna 5.0.0, a 16-point grid (<c>GridSampler({"x": [0.0, …, 15.0]}, seed=0)</c>) on a
    ///     study that already holds 12 added trials, then <c>optimize(n_trials=16)</c>: <c>x</c> = <c>3, 0, 5, 12, 6, 14,
    ///     7, 15, 10, 1, 9, 13, 11, 8, 2, 4</c>; the port's 13th value is <c>4</c> instead of <c>11</c>, once four
    ///     unvisited grids remain (seed 2 parts at the same trial, seed 1 happens to agree). Matching it means
    ///     reproducing CPython's set layout for small ints (hash = value, 8-slot minimum table, linear probing).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void GridSampler_ShiftedNumbering_PicksUnvisitedGridsInCPythonSetOrder()
    {
        var space = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.IReadOnlyList<OptunaSharp.Distributions.CategoricalChoice>>
        {
            ["x"] = System.Linq.Enumerable.ToList(System.Linq.Enumerable.Select(
                System.Linq.Enumerable.Range(0, 16), i => (OptunaSharp.Distributions.CategoricalChoice)(double)i)),
        };
        var study = Optuna.CreateStudy(sampler: new GridSampler(space, seed: 0));
        for (int i = 0; i < 12; i++)
            study.AddTrial(FrozenTrial.CreateTrial(value: i,
                @params: new OptunaSharp.Collections.OrderedDict<string, object?> { ["x"] = (double)i },
                distributions: new OptunaSharp.Collections.OrderedDict<string, OptunaSharp.Distributions.BaseDistribution>
                    { ["x"] = new OptunaSharp.Distributions.FloatDistribution(0, 15) }));
        study.Optimize(t => t.SuggestFloat("x", 0, 15), nTrials: 16);

        var xs = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(
            System.Linq.Enumerable.Skip(study.Trials, 12), t => (double)t.Params["x"]!));
        xs.Should().Equal(3.0, 0.0, 5.0, 12.0, 6.0, 14.0, 7.0, 15.0, 10.0, 1.0, 9.0, 13.0, 11.0, 8.0, 2.0, 4.0);
    }

    /// <summary>
    ///     QA-79: multi-objective TPE ranks the PRUNED trials it puts in its "below" group by their last intermediate
    ///     value. Optuna's <c>_get_pruned_trial_score</c> reads <c>study.direction</c> for that, and on a multi-objective
    ///     study the property raises <c>RuntimeError("A single direction cannot be retrieved from a multi-objective
    ///     study. Consider using Study.directions to retrieve a list containing all directions.")</c>, so the first
    ///     TPE trial fails. The port's <c>TpeTrialSplit.GetPrunedTrialScore</c> reads <c>study.Directions[0]</c>
    ///     instead, ranks the pruned trials by the first objective's direction and samples on.
    /// </summary>
    /// <remarks>
    ///     <c>Trial.report</c> refuses multi-objective studies on both sides, so the history comes from
    ///     <c>add_trial</c> here; a journal or RDB written by another tool can hold the same records. Measured with
    ///     Optuna 5.0.0: two objectives, <c>TPESampler(n_startup_trials=0, seed=0)</c>, one COMPLETE trial and four
    ///     PRUNED trials with intermediate values added, then <c>optimize(n_trials=12)</c> raises the RuntimeError
    ///     from the first trial's <c>suggest_float</c>; the port completes all 12 trials. The single-objective path is
    ///     unaffected (<c>study.direction</c> is defined there).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void MultiObjectiveTpe_PrunedTrialsWithIntermediateValues_RaisesLikeOptunasDirectionProperty()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new TPESampler(nStartupTrials: 0, seed: 0));
        var distributions = new OptunaSharp.Collections.OrderedDict<string, OptunaSharp.Distributions.BaseDistribution>
            { ["x"] = new OptunaSharp.Distributions.FloatDistribution(0, 1) };
        study.AddTrial(FrozenTrial.CreateTrial(values: new[] { 0.5, 0.5 },
            @params: new OptunaSharp.Collections.OrderedDict<string, object?> { ["x"] = 0.5 }, distributions: distributions));
        study.AddTrial(FrozenTrial.CreateTrial(state: TrialState.Pruned,
            @params: new OptunaSharp.Collections.OrderedDict<string, object?> { ["x"] = 0.2 }, distributions: distributions,
            intermediateValues: new OptunaSharp.Collections.OrderedDict<long, double> { [0] = 0.3, [1] = 0.1 }));
        for (int i = 0; i < 3; i++)
            study.AddTrial(FrozenTrial.CreateTrial(state: TrialState.Pruned,
                @params: new OptunaSharp.Collections.OrderedDict<string, object?> { ["x"] = 0.1 * i },
                distributions: distributions,
                intermediateValues: new OptunaSharp.Collections.OrderedDict<long, double> { [0] = 0.3 + i }));

        System.Action optimize = () => study.Optimize(t => new[] { t.SuggestFloat("x", 0, 1), 1 - t.SuggestFloat("x", 0, 1) },
            nTrials: 12);

        optimize.Should().Throw<NumSharp.RuntimeError>().WithMessage(
            "A single direction cannot be retrieved from a multi-objective study. Consider using Study.directions to " +
            "retrieve a list containing all directions.");
    }

    /// <summary>
    ///     QA-80: cmaes's <c>CMA</c> and <c>SepCMA</c> constructors assert <c>_is_valid_bounds(bounds, mean)</c>
    ///     ("invalid bounds"), so Optuna's first CMA-ES trial fails with <c>AssertionError</c> when the initial mean lies
    ///     outside the search space: an <c>x0</c> outside the ranges, or <c>source_trials</c> whose best solutions lie
    ///     outside the target's ranges (CmaEsSampler checks only parameter names). The port's <c>Cma</c> and
    ///     <c>SepCma</c> omit the assertion (only <c>CmaWm</c> keeps it, as <c>ArgumentException("invalid bounds")</c>),
    ///     so the study samples on around an out-of-bounds mean.
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0 / cmaes 0.13.1: <c>CmaEsSampler(x0={"x": 12.0, "y": 0.5}, seed=0)</c> over
    ///     <c>x</c> in [0, 10], <c>y</c> in [0, 1], <c>optimize(n_trials=5)</c>: trial 0 (independent) COMPLETE, trial 1
    ///     FAIL and <c>AssertionError: invalid bounds</c> raised, the same with <c>use_separable_cma=True</c> and
    ///     <c>lr_adapt=True</c>; <c>x0={"x": 10.0}</c> (on the bound) runs identically on both sides. The port completes
    ///     all five trials. The Warm-Starting fuzz (E51) hit it in 3 of 150 programs through <c>source_trials</c>
    ///     ranges. The expected exception type follows the port's own mapping of this assertion in <c>CmaWm</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CmaEs_InitialMeanOutsideTheBounds_FailsLikeCmaesInvalidBoundsAssertion()
    {
        foreach (bool separable in new[] { false, true })
        {
            var study = Optuna.CreateStudy(sampler: new CmaEsSampler(
                x0: new System.Collections.Generic.Dictionary<string, object?> { ["x"] = 12.0, ["y"] = 0.5 },
                seed: 0, useSeparableCma: separable));

            System.Action optimize = () => study.Optimize(t => t.SuggestFloat("x", 0, 10) + t.SuggestFloat("y", 0, 1),
                nTrials: 5);

            optimize.Should().Throw<System.ArgumentException>().WithMessage("invalid bounds", $"separable: {separable}");
            System.Linq.Enumerable.Select(study.Trials, t => t.State).Should().Equal(TrialState.Complete, TrialState.Fail);
        }
    }

    /// <summary>
    ///     QA-80 (second assertion): cmaes' <c>CMA</c> constructor also asserts its learning rates
    ///     (<c>c1 &lt;= 1 - cmu</c>, "invalid learning rate for the rank-one update", then <c>cmu</c>, <c>c_sigma</c>,
    ///     <c>cc</c>). With <c>popsize=1</c>, <c>mu = 0</c> makes <c>mu_eff</c> NaN, the comparison fails and Optuna's
    ///     first CMA-ES trial FAILs with <c>AssertionError</c>. The port's <c>Cma</c> has none of these assertions: it
    ///     builds the optimizer with NaN rates, completes trial 1, and fails trial 2 with
    ///     <c>ValueError("`nan` is invalid value.")</c> instead.
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0 / cmaes 0.13.1: <c>CmaEsSampler(seed=0, popsize=1)</c> over a float and an int,
    ///     <c>optimize(n_trials=8)</c>: states COMPLETE, FAIL and <c>AssertionError: invalid learning rate for the
    ///     rank-one update</c>, the same with <c>with_margin=True</c> and <c>lr_adapt=True</c>
    ///     (<c>use_separable_cma=True</c> raises <c>ZeroDivisionError</c>); the port gives COMPLETE, COMPLETE, FAIL in
    ///     all four. <c>popsize=2</c> is identical on both sides. The expected type follows the port's mapping of
    ///     cmaes assertions (<c>ArgumentException</c>, as for "invalid bounds" in <c>CmaWm</c>).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CmaEs_PopulationOfOne_FailsLikeCmaesLearningRateAssertion()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 0, popsize: 1));

        System.Action optimize = () => study.Optimize(t => t.SuggestFloat("x", 0, 10) + t.SuggestInt("y", 0, 5), nTrials: 8);

        optimize.Should().Throw<System.ArgumentException>().WithMessage("invalid learning rate for the rank-one update");
        System.Linq.Enumerable.Select(study.Trials, t => t.State).Should().Equal(TrialState.Complete, TrialState.Fail);
    }

    /// <summary>
    ///     QA-83: an <c>x0</c> that lacks one of the search space's parameters fails CMA-ES's first trial on both sides,
    ///     but with different errors. Optuna's <c>_SearchSpaceTransform.transform</c> asserts <c>name in params</c> with
    ///     the message "Parameter configuration must contain all distributions."; the port's transform indexes the
    ///     dictionary and lets <c>KeyNotFoundException</c> ("The given key 'y' was not present in the dictionary.")
    ///     escape, which names the missing key but not the cause.
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0: <c>CmaEsSampler(x0={"x": 5.0}, seed=0)</c> over <c>x</c> and <c>y</c>,
    ///     <c>optimize(n_trials=4)</c>: COMPLETE, FAIL and <c>AssertionError</c>; the port: COMPLETE, FAIL and the
    ///     <c>KeyNotFoundException</c>. An <c>x0</c> with an extra parameter is accepted and identical on both sides. The
    ///     expected type follows the port's mapping of Python assertions (<c>ArgumentException</c>, as in <c>CmaWm</c>).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CmaEs_X0MissingAParameter_FailsWithOptunasTransformAssertion()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(
            x0: new System.Collections.Generic.Dictionary<string, object?> { ["x"] = 5.0 }, seed: 0));

        System.Action optimize = () => study.Optimize(t => t.SuggestFloat("x", 0, 10) + t.SuggestFloat("y", 0, 1), nTrials: 4);

        optimize.Should().Throw<System.ArgumentException>().WithMessage("Parameter configuration must contain all distributions.");
        System.Linq.Enumerable.Select(study.Trials, t => t.State).Should().Equal(TrialState.Complete, TrialState.Fail);
    }

    /// <summary>
    ///     QA-84 (found 2026-11-11 by plan 15, after the run): Optuna sums a candidate's continuous log terms over the
    ///     dimensions with <c>_truncnorm.logpdf(...).sum(axis=-1)</c>, a reduction over a C-contiguous last axis, which
    ///     NumPy performs with its pairwise sum: from eight elements on, eight accumulators, then
    ///     <c>((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))</c>, then the remainder. The port added the terms left to
    ///     right from +0.0, so with eight or more continuous parameters many sums differed in their last bits.
    ///     <b>Fixed 2026-11-12</b> (owner's choice): the per-dimension terms are stored as NumPy's
    ///     <c>(n_samples, n_kernels, n_cont)</c> array and summed by NumSharp's <c>np.sum(axis=-1)</c>
    ///     (<c>MixtureOfProductDistribution.AddContinuousSums</c>).
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0's own <c>_truncnorm.logpdf</c> on its own array layout and NumPy 2.4.2: the
    ///     pairwise model below matches NumPy in all 7,200 (candidate, kernel) entries for 8, 10 and 16 dimensions, and a
    ///     left-to-right sum differed in 31–54% of them; NumSharp's <c>np.sum(axis=-1)</c> matches live NumPy at every
    ///     width from 1 to 300. The test isolates the order: both sides sum the port's own per-dimension terms, so only the
    ///     summation can differ.
    /// </remarks>
    [TestMethod]
    public void Tpe_ContinuousLogPdf_SumsTheDimensionsPairwiseLikeNumPy()
    {
        var rng = new System.Random(84);
        const int nSamples = 24, nCont = 10, nKernels = 300;
        int gridSize = nCont * nKernels;
        var mu = new double[gridSize];
        var sigma = new double[gridSize];
        var a = new double[gridSize];
        var b = new double[gridSize];
        var logSigma = new double[gridSize];
        for (int g = 0; g < gridSize; g++)
        {
            mu[g] = rng.NextDouble() * 20 - 10;
            sigma[g] = 0.2 + rng.NextDouble() * 5;
            a[g] = (-10 - mu[g]) / sigma[g];
            b[g] = (10 - mu[g]) / sigma[g];
            logSigma[g] = System.Math.Log(sigma[g]);
        }
        double[] logGaussMass = OptunaSharp.Internal.TruncNorm.LogGaussMass(a, b);
        var xCont = new double[nSamples * nCont];
        for (int i = 0; i < xCont.Length; i++)
            xCont[i] = rng.NextDouble() * 20 - 10;

        var actual = new double[nSamples * nKernels];
        OptunaSharp.Samplers.Tpe.MixtureOfProductDistribution.AddContinuousSums(
            xCont, nSamples, nCont, mu, sigma, a, b, logGaussMass, logSigma, actual, nKernels);

        int differing = 0;
        var terms = new double[nCont];
        for (int s = 0; s < nSamples; s++)
        {
            for (int k = 0; k < nKernels; k++)
            {
                for (int c = 0; c < nCont; c++)
                {
                    int g = c * nKernels + k;
                    double xstd = (xCont[s * nCont + c] - mu[g]) / sigma[g];
                    double val = OptunaSharp.Internal.TruncNorm.NormLogPdf(xstd) - logGaussMass[g] - logSigma[g];
                    if (a[g] == b[g])
                        val = double.NaN;
                    else if (xstd < a[g] || xstd > b[g])
                        val = double.NegativeInfinity;
                    terms[c] = val;
                }
                // weighted_log_pdf (zero here) += the row's NumPy sum.
                double expected = 0.0 + NumPyPairwiseSum(terms);
                if (System.BitConverter.DoubleToInt64Bits(actual[s * nKernels + k]) != System.BitConverter.DoubleToInt64Bits(expected))
                    differing++;
            }
        }

        differing.Should().Be(0, "NumPy sums each (candidate, kernel) row of the dimensions pairwise");
    }

    /// <summary>
    ///     NumPy's <c>pairwise_sum</c> for a contiguous run of at most 128 elements (one block): left to right from 0.0
    ///     below eight elements; from eight on, eight accumulators over whole groups of eight, combined as
    ///     <c>((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))</c>, then the remaining elements added left to right.
    /// </summary>
    /// <param name="x">The elements, at most 128 of them.</param>
    /// <returns>Their sum in NumPy's order.</returns>
    private static double NumPyPairwiseSum(double[] x)
    {
        int n = x.Length;
        if (n < 8)
        {
            double res0 = 0.0;
            for (int i = 0; i < n; i++)
                res0 += x[i];
            return res0;
        }
        var r = new double[8];
        System.Array.Copy(x, r, 8);
        int j = 8;
        for (; j < n - n % 8; j += 8)
        {
            for (int l = 0; l < 8; l++)
                r[l] += x[j + l];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; j < n; j++)
            res += x[j];
        return res;
    }
}

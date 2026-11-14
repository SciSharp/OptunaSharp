using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the samplers, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Every expected value is what Optuna 5.0.0 returns on
///     CPython 3.12 for the same seed and objective; each test fails today and passes once the divergence
///     is fixed, at which point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class SamplerParityBugTests
{
    /// <summary>
    ///     QA-11 (fixed 2026-11-06): <see cref="BruteForceSampler" /> must enumerate a stepped float range with
    ///     Python's correctly rounded <c>float(Decimal)</c>; the port's <c>(double)decimal</c> put 7% of candidate
    ///     values one ULP off whenever the lower bound is not a short decimal. It now converts through
    ///     <c>PyNumbers.DecimalToDouble</c>.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>_enumerate_candidates</c> walks <c>Decimal(str(low)) + k * Decimal(str(step))</c> and
    ///     converts each value with <c>float()</c>. Fed Python's own <c>(low, high, step)</c>, the port's
    ///     enumeration differs in 4,530 of 17,030 fuzzed ranges (60,839 of 856,670 values). Same root cause as
    ///     QA-2, different component: here it changes the parameter values a brute-force search tries. Python
    ///     5.0.0 with seed 0 tries exactly the nine values below; the port tries <c>7.607999463635719</c> in
    ///     place of <c>7.607999463635718</c>.
    /// </remarks>
    [TestMethod]
    public void BruteForce_SteppedFloatCandidates_MatchPythonDecimalEnumeration()
    {
        var study = Optuna.CreateStudy(sampler: new BruteForceSampler(seed: 0));
        study.Optimize(t => t.SuggestFloat("x", 3.6079994636357178, 11.607999463635718, step: 1.0), nTrials: 20);

        var tried = study.Trials.Select(t => PythonRepr.FloatRepr((double)t.Params["x"]!)).OrderBy(s => s, System.StringComparer.Ordinal).ToList();
        tried.Should().Equal(
            "10.607999463635718", "11.607999463635718", "3.6079994636357178", "4.607999463635718", "5.607999463635718",
            "6.607999463635718", "7.607999463635718", "8.607999463635718", "9.607999463635718");
    }

    /// <summary>
    ///     QA-23: constraint-violation totals must use CPython's compensated <c>sum()</c>, as Optuna's
    ///     <c>sum(v for v in trial.constraints.values() if v &gt; 0)</c> does; the port sums sequentially.
    /// </summary>
    /// <remarks>
    ///     CPython 3.12+ <c>sum()</c> over Python floats is Neumaier-compensated, so violations
    ///     <c>0.1, 0.2, 0.3</c> total <c>0.6</c>; the port's LINQ <c>Sum</c> gives <c>0.6000000000000001</c>. In
    ///     NSGA-II's <c>_constrained_dominates</c> the trial with those violations therefore dominates a trial
    ///     violating by <c>0.6000000000000001</c> in Python but not in the port, and <c>_evaluate_penalty</c>
    ///     ranks the two infeasible trials differently. The same sequential sum is in TPE's
    ///     <c>_get_infeasible_trial_score</c> (<c>TpeTrialSplit</c>), which orders infeasible trials for the
    ///     below/above split. Same root cause as the known importance-normalization item (catalog M7).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void ConstraintViolationSums_UseCompensatedSumLikeCPython()
    {
        var t0 = FrozenTrial.CreateTrial(values: new[] { 1.0 },
            constraints: new OrderedDict<string, double> { ["a"] = 0.1, ["b"] = 0.2, ["c"] = 0.3 });
        var t1 = FrozenTrial.CreateTrial(values: new[] { 1.0 },
            constraints: new OrderedDict<string, double> { ["a"] = 0.6000000000000001 });

        NsgaiiConstraintsEvaluation.ConstrainedDominates(t0, t1, new[] { StudyDirection.Minimize }).Should().BeTrue();
        NsgaiiConstraintsEvaluation.EvaluatePenalty(new[] { t0, t1 }).Select(PythonRepr.FloatRepr)
            .Should().Equal("0.6", "0.6000000000000001");
    }

    /// <summary>
    ///     QA-29: TPE must pick the candidate with <c>np.argmax</c>'s semantics, where the first NaN acquisition
    ///     value wins. The port's <c>ArgMax</c> skipped NaN until 2026-11-11; it now follows NumPy's rule (pinned by
    ///     <c>NumPyReductionOrderTests.ArgMaxPorts_FirstNanThenFirstMaximum_LikeNumPysArgmax</c>), but this run still
    ///     differs at trial 5 through the documented erf / UCRT tolerance (the last paragraph of the remarks).
    /// </summary>
    /// <remarks>
    ///     The acquisition value is <c>log l(x) - log g(x)</c>. When a candidate's probability mass is zero under
    ///     both mixtures it is <c>-inf - (-inf) = NaN</c>: for a log int above 2^53, <c>x - 0.5</c> and
    ///     <c>x + 0.5</c> are the same double, and for linear ranges of ~1e16 or more grid steps the mass falls
    ///     below the resolution of <c>1 - ndtr(a) - ndtr(-b)</c>. Optuna's <c>_compare</c> then returns the first
    ///     NaN candidate; the port returns the largest finite one. With <c>suggest_int("v", 1, 2**62, log=True)</c>
    ///     and TPE seed 3, 5 of the 24 candidates of the first TPE trial are NaN: Optuna picks index 5
    ///     (<c>235956546695320384</c>), the port index 11 (<c>114</c>). Replacing <c>np.argmax</c> with the port's
    ///     loop in Python reproduces <c>114</c>, which isolates the cause. Only the first six trials are compared,
    ///     because later ones also carry the documented erf ULP tolerance.
    ///     <para>
    ///         Since the argmax fix the port picks its first NaN, but its NaNs are not Optuna's: the candidates carry the
    ///         erf / UCRT ULP tolerance, and candidate 1 is <c>222479646760819</c> where Optuna draws
    ///         <c>222479646760813</c>. At that magnitude the bin <c>[log(x - 0.5), log(x + 0.5)]</c> spans about one
    ///         double, and for the port's value it collapses: both log densities are <c>-inf</c>, the acquisition is NaN
    ///         at index 1 (Optuna's is <c>-0.29165848147553675</c>), and the port picks <c>222479646760819</c> (measured
    ///         2026-11-11; Optuna's NaNs are at 5, 6, 8, 9 and 21, the port's at 1, 5, 6, 8, 9 and 21). The remaining
    ///         difference is the erf class, so this reproduction passes only once that tolerance is closed.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Tpe_NanAcquisitionValues_PickFirstNanLikeNumpyArgmax()
    {
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 3, nStartupTrials: 5));
        study.Optimize(t => t.SuggestInt("v", 1, 1L << 62, log: true) % 1000, nTrials: 6);

        study.Trials.Select(t => (long)t.Params["v"]!).Should().Equal(
            13956981037L, 13455981166018L, 164417L, 2436459554L, 43014068803425816L, 235956546695320384L);
    }

    /// <summary>
    ///     QA-31 (fixed 2026-11-06): in a batch of asked trials, <c>QMCSampler</c> must take the search space from the
    ///     trials of the same batch that already suggested their parameters, as Optuna does; the port sampled the
    ///     whole first batch independently, which also shifted the QMC sequence for the rest of the study. The
    ///     trial now mirrors its writes into the current thread's trial cache (<c>Trial.MirrorIntoThreadCache</c>).
    /// </summary>
    /// <remarks>
    ///     With no finished trial, <c>infer_relative_search_space</c> unions the distributions of the RUNNING trials
    ///     it reads from the study's per-ask trial cache. In Python that cache holds the storage's own trial objects,
    ///     and <c>Trial._suggest</c> mutates the same objects through <c>_cached_frozen_trial</c>, so trial 1 sees trial
    ///     0's <c>x</c> and gets Sobol point 0. The port's <see cref="Trial" /> keeps a detached copy (commit 3a75811,
    ///     added against a data race) and the cache keeps the old snapshot, so trials 1-3 see no distribution and are
    ///     sampled at random; QMC starts at trial 4 with sample id 0, so every later trial is Optuna's trial
    ///     <c>n - 3</c>. Optuna (seed 5, four asks, then four suggests and tells, then one more trial):
    ///     <c>-1.668040973461563, -3.0, 0.0, 1.5, -1.5</c>. The other samplers that read the cache (TPE, CMA-ES, GP,
    ///     the GA base) only look at finished trials, which do not change between two asks, so only QMC is affected.
    /// </remarks>
    [TestMethod]
    public void Qmc_BatchedAsk_UsesSearchSpaceOfRunningTrialsLikeOptuna()
    {
        var study = Optuna.CreateStudy(sampler: new QMCSampler(seed: 5));
        var batch = Enumerable.Range(0, 4).Select(_ => study.Ask()).ToList();
        var xs = batch.Select(t => t.SuggestFloat("x", -3, 3)).ToList();
        for (int k = 0; k < batch.Count; k++)
            study.Tell(batch[k], xs[k] * xs[k]);
        var last = study.Ask();
        double xl = last.SuggestFloat("x", -3, 3);
        study.Tell(last, xl * xl);

        study.Trials.Select(t => PythonRepr.FloatRepr((double)t.Params["x"]!))
            .Should().Equal("-1.668040973461563", "-3.0", "0.0", "1.5", "-1.5");
    }

    /// <summary>
    ///     QA-34 (fixed 2026-11-12 by <c>1e8aba6</c>, with run 2's QA-80): CMA-ES must reject a mean outside the
    ///     search-space bounds, as cmaes 0.13.1's <c>CMA.__init__</c> does with
    ///     <c>assert ... _is_valid_bounds(bounds, mean), "invalid bounds"</c>; the port's <c>Cma</c> and <c>SepCma</c>
    ///     constructors skipped the check (only <c>CmaWm</c> had it) and optimized from the outside mean.
    /// </summary>
    /// <remarks>
    ///     With <c>CmaEsSampler(seed=1, x0={"x": 5.0, "y": 0.0})</c> on <c>x, y ∈ [-1, 1]</c>, Optuna completes trial 0
    ///     (the independent start-up trial) and fails every later trial with
    ///     <c>AssertionError('invalid bounds')</c>, so <c>optimize</c> without <c>catch</c> stops at trial 1. The port
    ///     completes all four trials, sampling around a mean the user could not have meant. <c>x0</c> is deprecated
    ///     since 4.9, but still accepted.
    /// </remarks>
    [TestMethod]
    public void CmaEs_X0OutsideBounds_FailsTrialsLikeCmaes()
    {
#pragma warning disable CS0618 // x0 is deprecated in Optuna 4.9, as in the port; the test needs it.
        var sampler = new CmaEsSampler(seed: 1, x0: new Dictionary<string, object?> { ["x"] = 5.0, ["y"] = 0.0 });
#pragma warning restore CS0618
        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(
            t => Math.Pow(t.SuggestFloat("x", -1, 1) - 0.2, 2) + Math.Pow(t.SuggestFloat("y", -1, 1), 2),
            nTrials: 4, @catch: new[] { typeof(Exception) });

        study.Trials.Select(t => t.State).Should().Equal(
            TrialState.Complete, TrialState.Fail, TrialState.Fail, TrialState.Fail);
    }

    /// <summary>
    ///     QA-41 (fixed upstream in NumSharp 0.80.0): sampling a float range whose span is not finite must fail without
    ///     consuming the random stream, as numpy's <c>RandomState.uniform</c> does with
    ///     <c>OverflowError('Range exceeds valid bounds')</c>; the port drew anyway, returned <c>inf</c> for
    ///     <c>[1e-3, inf)</c> and shifted every later random value of the study, until NumSharp's legacy
    ///     <c>RandomState</c> became NumPy's <c>legacy-distributions.c</c> line for line (NumSharp <c>f87b4d04</c>).
    /// </summary>
    /// <remarks>
    ///     <c>FloatDistribution</c> accepts infinite bounds in both languages, so the failure comes from the sampler:
    ///     <c>RandomSampler.sample_independent</c> calls <c>rng.uniform(bounds[:, 0], bounds[:, 1])</c>, and numpy checks
    ///     <c>np.all(np.isfinite(high - low))</c> before drawing. NumSharp's <c>NumPyRandom.uniform(NDArray, NDArray)</c>
    ///     computes <c>low + rand * (high - low)</c> without the check, so the port stores <c>inf</c> as the parameter,
    ///     <c>(-inf, 1)</c> ends in <c>ValueError: `nan` is invalid value.</c>, and a span that overflows such as
    ///     <c>(-1e308, 1e308)</c> is sampled too. Every sampler's independent path goes through it (TPE start-up, QMC,
    ///     NSGA-II, CMA-ES all raise <c>OverflowError</c> in Optuna), and on a finished trial Optuna reports the
    ///     <c>OverflowError</c> before <c>UpdateFinishedTrialError</c> because it samples first. Python (Random seed 0):
    ///     the failed suggestion leaves the stream untouched, so the next <c>suggest_float("y", 0, 1)</c> is
    ///     <c>0x3fe18fe1565f12a8</c>; the port returns <c>0x3fe6e2d4cf608733</c>, the second draw. Found by the D79 API
    ///     state-machine fuzz.
    /// </remarks>
    [TestMethod]
    public void RandomSampler_InfiniteRange_RaisesOverflowWithoutDrawingLikeNumpyUniform()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        var trial = study.Ask();

        // NumPy's class: NumSharp.OverflowError since NumSharp 0.81.0 (it was .NET's OverflowException before).
        ((Action)(() => trial.SuggestFloat("x", 1e-3, double.PositiveInfinity))).Should().ThrowExactly<NumSharp.OverflowError>()
            .WithMessage("Range exceeds valid bounds");
        ((Action)(() => trial.SuggestFloat("w", -1e308, 1e308))).Should().ThrowExactly<NumSharp.OverflowError>()
            .WithMessage("Range exceeds valid bounds");
        BitConverter.DoubleToInt64Bits(trial.SuggestFloat("y", 0, 1)).Should().Be(0x3fe18fe1565f12a8);
    }

    /// <summary>
    ///     QA-42 (root cause; fixed upstream in NumSharp 0.80.0): <c>randint(low, low + 1)</c> must return <c>low</c>
    ///     without consuming the stream, as numpy's bounded-integer generator does (<c>random_bounded_uint64_fill</c>
    ///     returns <c>off</c> when the range is 0); NumSharp's <c>MT19937.Next(1)</c> drew a 32-bit word and masked it to
    ///     0 until its legacy <c>RandomState</c> rewrite (NumSharp <c>f87b4d04</c>).
    /// </summary>
    /// <remarks>
    ///     numpy 2.4.2, <c>RandomState(0)</c>: after <c>randint(0, 1)</c> (or <c>choice(1)</c>) the next
    ///     <c>random_sample()</c> is <c>0x3fe18fe1565f12a8</c>, the stream's first double; NumSharp returns
    ///     <c>0x3fe2f89546dc5a99</c>, the value numpy gives only after a draw that consumed one word (such as
    ///     <c>randint(0, 2)</c>). Every OptunaSharp call of <c>randint</c> / <c>choice</c> over a single candidate shifts
    ///     the rest of its sampler's stream. The fix belongs in NumSharp (<c>RandomSampling/MT19937.cs</c>,
    ///     <c>Next(int)</c> / <c>NextLong(long)</c>), or in the call sites.
    /// </remarks>
    [TestMethod]
    public void RandInt_SingleValueRange_DrawsNothingLikeNumpy()
    {
        var rng = new LazyRandomState(0).Rng;

        ((long)rng.randint(0, 1)).Should().Be(0);
        BitConverter.DoubleToInt64Bits((double)rng.random_sample()).Should().Be(0x3fe18fe1565f12a8);
    }

    /// <summary>
    ///     QA-42 (fixed upstream in NumSharp 0.80.0): NSGA-II's parent tournament over a single remaining candidate must
    ///     leave the random stream untouched, as Optuna's <c>_select_parent</c> does (<c>rng.choice(1)</c> draws nothing);
    ///     the port consumed a word per call, so with <c>population_size=2</c> every child after the first generation
    ///     differed.
    /// </summary>
    /// <remarks>
    ///     <c>_select_parents</c> excludes the parents already chosen, so with a population of 2 the second parent is
    ///     drawn from one candidate with two <c>choice(1)</c> calls; the same happens with a population of 3 under the
    ///     three-parent SPX / UNDX crossovers, in NSGA-III, and whenever failed trials leave a parent population of that
    ///     size. A one-parameter study hides it because the default mutation hands every parameter to the independent
    ///     sampler, whose stream is separate, so the reproduction uses two parameters and <c>mutation_prob=0.0</c>.
    ///     Optuna 5.0.0 (seed 0, trials 2–5): <c>x = 0.0976270078546495, 0.20552675214328775 ×3</c>,
    ///     <c>y = 0.43037873274483895 ×4</c>; the port gives <c>y = 0.08976636599379373</c>. Found by the D82
    ///     sampler-option fuzz (20 of 600 programs, all with population 2 or 3).
    /// </remarks>
    [TestMethod]
    public void NsgaII_PopulationOfTwo_SecondParentDrawLeavesStreamLikeOptuna()
    {
        var study = Optuna.CreateStudy(sampler: new NSGAIISampler(populationSize: 2, mutationProb: 0.0, seed: 0));
        study.Optimize(
            t => Math.Pow(t.SuggestFloat("x", -1, 1) - 0.25, 2) + Math.Pow(t.SuggestFloat("y", -1, 1), 2), nTrials: 6);

        study.Trials.Skip(2).Select(t => PythonRepr.FloatRepr((double)t.Params["x"]!))
            .Should().Equal("0.0976270078546495", "0.20552675214328775", "0.20552675214328775", "0.20552675214328775");
        study.Trials.Skip(2).Select(t => PythonRepr.FloatRepr((double)t.Params["y"]!))
            .Should().Equal("0.43037873274483895", "0.43037873274483895", "0.43037873274483895", "0.43037873274483895");
    }

    /// <summary>
    ///     QA-43 (GridSampler): the grid must be enumerated over the parameter names in Python's <c>sorted()</c> order,
    ///     which compares code points; the port sorts with <see cref="StringComparer.Ordinal" /> (UTF-16 code units), so
    ///     the same seed visits the grid in a different order when one name has a character above U+FFFF and another one
    ///     in U+E000–U+FFFF.
    /// </summary>
    /// <remarks>
    ///     <c>GridSampler.__init__</c> keeps <c>sorted(search_space.items())</c> and builds
    ///     <c>itertools.product</c> over it, so the order decides which combination each grid id means. With
    ///     <c>ｘ</c> (U+FF58) and <c>😀</c> (U+1F600) Python sorts <c>ｘ</c> first and the port <c>😀</c> first (its leading
    ///     code unit U+D83D is below U+FF58). Optuna 5.0.0, seed 0, six trials, as <c>(ｘ, 😀)</c>:
    ///     <c>(2,5) (1,5) (1,4) (2,3) (1,3) (2,4)</c>; the port visits <c>(2,5) (1,4) (2,3) (2,4) (1,3) (1,5)</c>.
    ///     Found by the D85 sort-order probe; see the dataframe test for the other sites.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void GridSampler_AstralCharacterNames_EnumerateGridLikePython()
    {
        string fullwidthX = "ｘ", grinning = char.ConvertFromUtf32(0x1F600);
        var sampler = new GridSampler(new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            [fullwidthX] = new CategoricalChoice[] { 1L, 2L },
            [grinning] = new CategoricalChoice[] { 3L, 4L, 5L },
        }, seed: 0);
        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(t => t.SuggestInt(fullwidthX, 1, 2) * 10 + t.SuggestInt(grinning, 3, 5), nTrials: 6);

        study.Trials.Select(t => $"({t.Params[fullwidthX]},{t.Params[grinning]})")
            .Should().Equal("(2,5)", "(1,5)", "(1,4)", "(2,3)", "(1,3)", "(2,4)");
    }

    /// <summary>
    ///     QA-48 (a) (fixed 2026-11-11): the TPE <c>weights</c> validation errors must print the rejected array the way
    ///     NumPy's <c>str(ndarray)</c> does (<c>[ 0.5  -0.25]</c>, <c>[0.]</c>). The port joined Python float reprs with
    ///     one space (<c>[0.5 -0.25]</c>, <c>[0.0]</c>); it now prints with NumSharp's <c>NDArray.ToString()</c>, a port of
    ///     NumPy's array printer.
    /// </summary>
    /// <remarks>
    ///     <c>_ParzenEstimator._call_weights_func</c> formats the array with an f-string, i.e. NumPy's array printer:
    ///     a common field width, a sign column when any element is negative, and <c>1.</c> rather than <c>1.0</c>.
    ///     The port's <c>CallWeightsFunc</c> says in a comment that it formats "approximately"; nothing records the
    ///     divergence as <c>[Misaligned]</c>. Only the deprecated custom <c>weights</c> argument reaches it. With two
    ///     startup trials the first below group has one observation and the first rejected above group two, so
    ///     Optuna 5.0.0 raises the two messages below (found by the D92 TPE option probe).
    /// </remarks>
    [TestMethod]
    public void Tpe_InvalidWeights_MessagePrintsArrayLikeNumpy()
    {
        static double Objective(Trial t) =>
            new Dictionary<string, double> { ["a"] = 1.0, ["b"] = 2.0, ["c"] = 3.0 }[t.SuggestCategorical("x", new[] { "a", "b", "c" })];

        var mixed = Optuna.CreateStudy(sampler: new TPESampler(seed: 0, nStartupTrials: 2,
            weights: n => n <= 5 ? new[] { 0.5, -0.25, 1e-7, 3.0, 12345.678 }[..n] : Enumerable.Repeat(-1.0, n).ToArray()));
        ((Action)(() => mixed.Optimize(Objective, nTrials: 5))).Should().Throw<NumSharp.ValueError>().Which.Message.Should().Be(
            "The `weights` function is not allowed to return negative values [ 0.5  -0.25]. The argument of the `weights` function is 2.");

        var zeros = Optuna.CreateStudy(sampler: new TPESampler(seed: 0, nStartupTrials: 2, weights: n => new double[n]));
        ((Action)(() => zeros.Optimize(Objective, nTrials: 5))).Should().Throw<NumSharp.ValueError>().Which.Message.Should().Be(
            "The `weight` function is not allowed to return all-zero values [0.]. The argument of the `weights` function is 1.");
    }

    /// <summary>
    ///     QA-48 (b): a TPE <c>weights</c> function that returns fewer values than observations must fail the first
    ///     TPE trial with a <c>ValueError</c>, as NumPy's broadcasting does in Optuna; the port pads nothing and
    ///     silently drops the prior kernel from the mixture.
    /// </summary>
    /// <remarks>
    ///     <c>_call_weights_func</c> truncates a longer array (<c>[:n]</c>) but accepts a shorter one; Optuna then fails
    ///     in <c>_MixtureOfProductDistribution</c> with <c>ValueError: non-broadcastable output operand with shape (24,1)
    ///     doesn't match the broadcast shape (24,2)</c>. The port appends the prior weight to the short array, so the
    ///     weight vector has one entry fewer than the kernels and the last kernel (the prior) is never drawn nor
    ///     scored. The NumPy message itself is not asserted; any <c>ValueError</c> would match Optuna's outcome.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Tpe_ShortWeights_RaisesLikeNumpyBroadcast()
    {
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 0, nStartupTrials: 2,
            weights: n => Enumerable.Repeat(1.0, Math.Max(n - 1, 0)).ToArray()));
        ((Action)(() => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 5))).Should().Throw<NumSharp.ValueError>();
    }

    /// <summary>
    ///     QA-48 (c): a NaN <c>prior_weight</c> must raise NumPy's <c>ValueError("probabilities contain NaN")</c> at the
    ///     first TPE sample; the port's hand-written <c>rng.choice</c> skips NumPy's <c>p</c> validation and walks off
    ///     the end of its CDF with an <see cref="IndexOutOfRangeException" />, which <c>optimize(catch: ValueError)</c>
    ///     does not catch.
    /// </summary>
    /// <remarks>
    ///     <c>_ParzenEstimator</c> only rejects <c>prior_weight &lt; 0</c>, which NaN passes; the normalized weights are
    ///     then all NaN and <c>RandomState.choice(p=weights)</c> raises before drawing. The port's
    ///     <c>MixtureOfProductDistribution.Choice</c> (replicated because NumSharp's <c>choice</c> uses
    ///     <c>side='left'</c>) bisects a NaN CDF and indexes one past its end. The same missing validation turns
    ///     <c>prior_weight=inf</c> (all weights <c>0 / inf</c> and <c>inf / inf</c>) into the same crash.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Tpe_NaNPriorWeight_RaisesProbabilitiesContainNaNLikeNumpy()
    {
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 0, nStartupTrials: 2, priorWeight: double.NaN));
        ((Action)(() => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 5)))
            .Should().Throw<NumSharp.ValueError>().WithMessage("probabilities contain NaN");
    }

    /// <summary>
    ///     QA-48 (d): with a <c>gamma</c> that returns a negative count, a single-objective TPE must split the trials
    ///     with Python slice semantics (<c>sorted[:-1]</c> keeps every trial but the worst below); the port's
    ///     <c>Take(-1)</c> keeps none, so the trajectories part at the first TPE trial.
    /// </summary>
    /// <remarks>
    ///     <c>_split_complete_trials</c> clamps only from above (<c>min(n_below, len(trials))</c>) and slices, and the
    ///     later <c>max(0, …)</c> clamps apply to the pruned and infeasible groups only. The multi-objective path
    ///     raises <c>AssertionError</c> instead (QA-16). Only the deprecated custom <c>gamma</c> argument reaches it.
    ///     Optuna 5.0.0, seed 0, two startup trials, <c>gamma = lambda n: -1</c>: the TPE trials all exploit
    ///     <c>(b, d)</c>; the port behaves as <c>gamma = 0</c> (<c>da cc ca dc …</c>). Found by the D92 option fuzz:
    ///     the 8 programs with a negative gamma are the only divergences in 600.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Tpe_NegativeGammaSingleObjective_SplitsWithPythonSliceSemantics()
    {
        var val = new Dictionary<string, double> { ["a"] = 1.0, ["b"] = 2.0, ["c"] = 3.0, ["d"] = 0.5 };
        var choices = new[] { "a", "b", "c", "d" };
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 0, nStartupTrials: 2, multivariate: true, gamma: _ => -1));
        study.Optimize(t => val[t.SuggestCategorical("x", choices)] + val[t.SuggestCategorical("y", choices)], nTrials: 12);

        study.Trials.Select(t => (string)t.Params["x"]! + (string)t.Params["y"]!)
            .Should().Equal("bd", "ab", "bd", "bd", "bd", "bd", "bd", "bd", "bd", "bd", "bd", "bd");
    }

    /// <summary>
    ///     QA-49 (fixed 2026-11-11): the GP search space must build a stepped parameter's grid exactly as
    ///     <c>np.arange(low, high + step / 2, step)</c> does — every value <c>low + i * ((low + step) - low)</c>. The port's
    ///     <c>GpSearchSpace.Arange</c> computed <c>low + i * step</c>, so the normalized grid differed in its last bits
    ///     whenever <c>(low + step) - low != step</c>; it now calls NumSharp's <c>np.arange</c>, which fills NumPy's way.
    /// </summary>
    /// <remarks>
    ///     NumPy's <c>arange</c> stores <c>start</c> and <c>start + step</c> and fills the rest with
    ///     <c>start + i * delta</c>, <c>delta = buffer[1] - buffer[0]</c>; for <c>FloatDistribution(0.1, 1.0, step=0.3)</c>
    ///     that is <c>[0.1, 0.4, 0.7000000000000001, 1.0000000000000002]</c>, against the port's
    ///     <c>[0.1, 0.4, 0.7, 0.9999999999999999]</c>. The two differ in 9,641 of 20,000 random GP-style ranges. The CMA-ES
    ///     port already copies this step (<c>CmaWm</c>: "with its actual step (low + step) - low"); the GP one does not.
    ///     Optuna 5.0.0's <c>get_choices_of_discrete_params()</c> for that distribution is the grid asserted below.
    /// </remarks>
    [TestMethod]
    public void GpSearchSpace_SteppedFloatGrid_UsesNumpyArangeStep()
    {
        var space = new OptunaSharp.Gp.GpSearchSpace(new Dictionary<string, BaseDistribution>
        {
            ["x"] = new FloatDistribution(0.1, 1.0, step: 0.3),
        });

        space.GetChoicesOfDiscreteParams()[0].Select(PythonRepr.FloatRepr)
            .Should().Equal("0.125", "0.375", "0.625", "0.8750000000000002");
    }

    /// <summary>
    ///     QA-49, the user-visible effect (fixed 2026-11-11): <see cref="GPSampler" /> suggests a stepped float by
    ///     unnormalizing the grid point it picked, without rounding it again, so the grid reaches <c>trial.params</c>; the
    ///     former <c>low + i * step</c> grid gave <c>0.9999999999999998</c> where Optuna suggests <c>1.0</c>.
    /// </summary>
    /// <remarks>
    ///     <c>_get_unnormalized_param</c> only clips a float to <c>[low, high]</c>; the grid value
    ///     <c>1.0000000000000002</c> clips to <c>1.0</c> in Optuna, while the port's <c>0.9999999999999999</c> comes back
    ///     from the normalize / unnormalize round trip as <c>0.9999999999999998</c>. Optuna 5.0.0, seed 0, two startup
    ///     trials, <c>(x - 0.7) ** 2</c> over <c>[0.1, 1.0]</c> step <c>0.3</c>; the port matches every trial but the fourth.
    ///     (With <c>[0.1, 2.5]</c> the same run suggests <c>1.9000000000000004</c> in Optuna and <c>1.9000000000000001</c> in
    ///     the port.)
    ///     <para>
    ///         The expected trials are Optuna's on x64, so the test is pinned to x64, as the suite's other host-dependent
    ///         bit-exact oracles are (AGE-MOEA-2's operators, the CMA-ES state oracle, sklearn, SHAP, the ES digests). CI's
    ///         first macOS arm64 run of it (2026-11-14, net8.0) suggested <c>0.9999999999999998</c> for the fourth trial, the
    ///         former grid's value, with the same choice in every trial: the stepped grid itself is NumPy's there too
    ///         (<see cref="GpSearchSpace_SteppedFloatGrid_UsesNumpyArangeStep" /> passed), so the GP's last-bit arithmetic on
    ///         arm64 reached that grid point by another route.
    ///     </para>
    /// </remarks>
    [TestMethod]
    public void GpSampler_SteppedFloat_SuggestsNumpyGridValues()
    {
        // x64 only: the expected values are an x64 Optuna run's last bits (see remarks).
        if (System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture != System.Runtime.InteropServices.Architecture.X64)
            Assert.Inconclusive(
                "The expected trials are Optuna 5.0.0's on x64; the GP's arithmetic differs in its last bits on " +
                $"{System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}, which moves the fourth trial's value.");

        var study = Optuna.CreateStudy(sampler: new GPSampler(seed: 0, nStartupTrials: 2));
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", 0.1, 1.0, step: 0.3) - 0.7, 2), nTrials: 8);

        study.Trials.Select(t => PythonRepr.FloatRepr((double)t.Params["x"]!))
            .Should().Equal("0.7", "0.7", "0.1", "1.0", "0.39999999999999997", "0.7", "0.7", "0.7");
    }

    /// <summary>
    ///     The GP search space's discrete rounding, <c>_round_one_normalized_param</c>, takes
    ///     <c>(param_value - low + 0.5 * step) // step</c>: NumPy's <c>floor_divide</c> floors the EXACT quotient, so a point
    ///     on a grid midpoint rounds down where <c>Math.Floor(a / b)</c> (the port until 2026-11-11) rounded up a whole
    ///     grid step. The Sobol'-style dyadic point <c>0.75</c> on <c>[-7.82, -6.67]</c> step <c>0.05</c> unnormalizes to the
    ///     midpoint <c>-6.945</c>; <c>a = 0.9</c> and <c>0.9 // 0.05</c> is <c>17</c> (the double <c>0.05</c> is a little over
    ///     1/20), but <c>0.9 / 0.05</c> rounds to <c>18.0</c>. Optuna 5.0.0 / NumPy 2.4.2 return the normalized
    ///     <c>0.7291666666666661</c> (grid point <c>-6.970000000000001</c>); the former rounding gave <c>0.7708333333333333</c>
    ///     (<c>-6.92</c>). About 1 in 800 random or dyadic points on stepped grids hit such a case.
    /// </summary>
    [TestMethod]
    public void GpSearchSpace_RoundingAGridMidpoint_FloorsTheExactQuotientLikeNumpy()
    {
        double rounded = OptunaSharp.Gp.GpSearchSpace.RoundOneNormalizedParam(
            0.75, OptunaSharp.Gp.GpSearchSpace.ScaleType.Linear, (-7.82, -6.67), 0.05);

        BitConverter.DoubleToInt64Bits(rounded).Should().Be(0x3FE7555555555550L,
            $"NumPy's floor_divide puts 0.75 on grid point 17 (0.7291666666666661), not {rounded:R}");
    }

    /// <summary>
    ///     QA-53 (fixed 2026-11-11): when every GP acquisition value is NaN, Optuna's discrete line search hands scipy's
    ///     Brent minimizer the bracket <c>(inf, NaN, inf)</c> and fails the suggestion with scipy's bracket
    ///     <c>ValueError</c>; the port returned a parameter value (<c>16</c> here) until <c>GPSampler.ColumnMax</c> became
    ///     NumSharp's NaN-propagating <c>np.amax</c>.
    /// </summary>
    /// <remarks>
    ///     Objective values <c>1.7e308, 1.7e308, -1.7e308</c> overflow <c>_standardize_values</c> (the mean is
    ///     <c>inf</c>, every standardized value NaN), so the acquisition is NaN at every candidate (instrumented in
    ///     Optuna: 2 of 2 start points NaN). For an <c>IntDistribution(0, 20)</c> (21 grid points, more than the 16 of
    ///     the exhaustive search) <c>_discrete_line_search</c> calls <c>minimize_scalar(method="brent")</c>, whose bracket
    ///     check raises the message below — its own comment says the bracket is valid only "if fval is finite". The
    ///     port's <c>BrentMinimizer.MinimizeScalar</c> has the same check, but the NaN never reaches it: the LogEI
    ///     threshold is <c>standardized_score_vals[:, 0].max()</c>, which NumPy returns as NaN when the column holds a
    ///     NaN, while the port's <c>GPSampler.ColumnMax</c> skips NaNs (<c>if (v &gt; max)</c> from <c>-inf</c>) and returns
    ///     <c>-inf</c>, so its acquisition stays finite. Root cause located by the 2026-11-10 run (E12): patching Optuna's
    ///     <c>LogEI</c> to receive <c>-inf</c> for a NaN threshold makes Optuna return <c>16</c> too. With a float
    ///     parameter both sides return the same point (<c>0.7903830111026764</c>). Exotic input (values near the float
    ///     maximum); found by the D94 probe.
    /// </remarks>
    [TestMethod]
    public void GpSampler_AllNaNAcquisition_IntLineSearchRaisesBracketErrorLikeScipy()
    {
        var study = Optuna.CreateStudy(sampler: new GPSampler(seed: 0, nStartupTrials: 3));
        var distribution = new IntDistribution(0, 20);
        var xs = new[] { 1L, 10L, 19L };
        var values = new[] { 1.7e308, 1.7e308, -1.7e308 };
        for (int i = 0; i < 3; i++)
            study.AddTrial(FrozenTrial.CreateTrial(@params: new OrderedDict<string, object?> { ["x"] = xs[i] },
                distributions: new OrderedDict<string, BaseDistribution> { ["x"] = distribution }, value: values[i]));

        var trial = study.Ask();
        ((Action)(() => trial.SuggestInt("x", 0, 20))).Should().Throw<ArgumentException>().WithMessage(
            "Bracketing values (xa, xb, xc) do not fulfill this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))");
    }

    /// <summary>
    ///     QA-55 (fixed 2026-11-11): a <see cref="GridSampler" /> whose search space has an empty value list must fail the
    ///     first trial with NumPy's <c>choice</c> error, <c>'a' cannot be empty unless no samples are taken</c>; the port's
    ///     shortcut <c>target_grids[randint(0, len)]</c> failed with NumSharp's <c>randint</c> message <c>low &gt;= high</c>
    ///     until it became NumSharp's own <c>RandomState.choice</c>.
    /// </summary>
    /// <remarks>
    ///     Optuna validates no grid's length, so <c>{"x": []}</c> builds an empty grid; <c>sample_relative</c> finds no
    ///     unvisited grid id, falls back to <c>list(range(len(self._all_grids)))</c> (also empty) and calls
    ///     <c>self._rng.rng.choice(target_grids)</c>. <c>GridSampler.SampleRelative</c> replaces <c>choice</c> with an
    ///     equivalent <c>randint</c> draw (same stream for non-empty lists), so only this error text differs. Both raise
    ///     <c>ValueError</c>. Found by the D112 grid probe; the other cases there (empty space, duplicate values, NaN, None,
    ///     unused and missing parameters, <c>True</c> / <c>1</c> choices) match Optuna 5.0.0.
    /// </remarks>
    [TestMethod]
    public void GridSampler_EmptyValueList_FailsWithNumpyChoiceMessage()
    {
        var study = Optuna.CreateStudy(sampler: new GridSampler(
            new Dictionary<string, IReadOnlyList<CategoricalChoice>> { ["x"] = System.Array.Empty<CategoricalChoice>() }, seed: 0));

        ((Action)(() => study.Optimize(_ => 0.0, nTrials: 1))).Should().Throw<NumSharp.ValueError>()
            .Which.Message.Should().Be("'a' cannot be empty unless no samples are taken");
    }
}

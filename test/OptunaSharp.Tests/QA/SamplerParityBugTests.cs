using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
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
    ///     QA-11: <see cref="BruteForceSampler" /> must enumerate a stepped float range with Python's correctly
    ///     rounded <c>float(Decimal)</c>; the port's <c>(double)decimal</c> puts 7% of candidate values one ULP
    ///     off whenever the lower bound is not a short decimal.
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
    [OpenBugs]
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
    ///     value wins; the port's <c>ArgMax</c> skips NaN (its comment assumes the values are finite).
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
    ///     QA-31: in a batch of asked trials, <c>QMCSampler</c> must take the search space from the trials of the same
    ///     batch that already suggested their parameters, as Optuna does; the port samples the whole first batch
    ///     independently, which also shifts the QMC sequence for the rest of the study.
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
    [OpenBugs]
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
}

using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna deep-parity gate for the DocFX <b>Recipes</b> pages
///     (<c>docs/website-src/docs/recipes/</c>). Each recipe's example mechanism is exercised here with a
///     concrete, deterministic objective and reproduced bit-for-bit against real Optuna: the OptunaSharp
///     study is captured with <see cref="StudySnapshot.Capture" />, the equivalent Python is run in the
///     embedded interpreter, and <see cref="StudySnapshot.AssertParity" /> compares every trial's number,
///     state, values, params (external + internal repr), distributions, user/system attributes and
///     intermediate values.
///
///     <para>Recipes whose point is not a specific sampler are driven by the RNG-exact
///     <see cref="RandomSampler" /> (bit-identical to NumPy's <c>RandomState</c>); recipes that showcase a
///     particular sampler use that sampler. This file covers the study-mechanism recipes; the
///     user-defined sampler/pruner, Wilcoxon, artifacts and CLI recipes have their own parity files.</para>
/// </summary>
[TestClass]
public class RecipeParityTests : OptunaInteropTestBase
{
    // ---------------------------------------------------------------- attributes.md

    /// <summary>Deterministic objective that also records a per-trial user attribute (the recipe's pattern).</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The squared error, minimized.</returns>
    private static double AttributesObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        double err = (x - 2) * (x - 2);
        t.SetUserAttr("squared_error", err);   // per-trial metadata, mirrored on the Python side
        return err;
    }

    private const string PyAttributesObjective =
        "def _obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    err = (x - 2) ** 2\n" +
        "    t.set_user_attr('squared_error', err)\n" +
        "    return err\n";

    /// <summary>
    ///     User Attributes recipe: study-level attributes (a list and a string) plus a per-trial attribute
    ///     set inside the objective must round-trip and match live Optuna exactly.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Attributes_StudyAndTrial_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "attrs");
        cs.SetUserAttr("contributors", new object?[] { "Akiba", "Sano" });
        cs.SetUserAttr("dataset", "MNIST");
        cs.Optimize(AttributesObjective, nTrials: 8);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='attrs', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.set_user_attr('contributors', ['Akiba', 'Sano'])\n" +
            "study.set_user_attr('dataset', 'MNIST')\n" +
            PyAttributesObjective +
            "study.optimize(_obj, n_trials=8)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ---------------------------------------------------------------- callback.md

    /// <summary>Objective that prunes every trial after the first five, so the stop-callback can fire.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>A suggested float for the first five trials; raises for the rest.</returns>
    private static double CallbackObjective(Trial t)
    {
        if (t.Number > 4)
            throw new TrialPruned();
        return t.SuggestFloat("x", 0, 1);
    }

    /// <summary>Stateful callback that stops the study after <c>threshold</c> consecutive pruned trials.</summary>
    private sealed class StopWhenTrialKeepBeingPrunedCallback
    {
        private readonly int _threshold;
        private int _consecutivePrunedCount;

        /// <param name="threshold">Consecutive-prune count that triggers the stop.</param>
        public StopWhenTrialKeepBeingPrunedCallback(int threshold) => _threshold = threshold;

        /// <summary>Invoked after each trial; stops the study once the prune streak reaches the threshold.</summary>
        /// <param name="study">The running study.</param>
        /// <param name="trial">The finished trial.</param>
        public void Invoke(Study study, FrozenTrial trial)
        {
            _consecutivePrunedCount = trial.State == TrialState.Pruned ? _consecutivePrunedCount + 1 : 0;
            if (_consecutivePrunedCount >= _threshold)
                study.Stop();
        }
    }

    /// <summary>
    ///     Callback recipe: a stop-after-two-consecutive-pruned callback must end the study at exactly 7
    ///     trials (5 complete + 2 pruned) on both runtimes, with identical trial state and params.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Callback_StopAfterConsecutivePruned_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "cb");
        var cb = new StopWhenTrialKeepBeingPrunedCallback(2);
        cs.Optimize(CallbackObjective, nTrials: 10, callbacks: new Action<Study, FrozenTrial>[] { cb.Invoke });
        var csState = StudySnapshot.Capture(cs);

        cs.Trials.Count.Should().Be(7, "the study stops after two consecutive pruned trials (5 complete + 2 pruned)");

        PyExec(
            "class _Cb:\n" +
            "    def __init__(self, threshold):\n" +
            "        self.threshold = threshold\n" +
            "        self._count = 0\n" +
            "    def __call__(self, study, trial):\n" +
            "        self._count = self._count + 1 if trial.state == optuna.trial.TrialState.PRUNED else 0\n" +
            "        if self._count >= self.threshold:\n" +
            "            study.stop()\n" +
            "def _obj(t):\n" +
            "    if t.number > 4:\n" +
            "        raise optuna.TrialPruned()\n" +
            "    return t.suggest_float('x', 0, 1)\n" +
            $"study = optuna.create_study(study_name='cb', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=10, callbacks=[_Cb(2)])\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ---------------------------------------------------------------- specify-params.md

    /// <summary>Deterministic three-parameter objective shared by the enqueue/add-trial recipe.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>A deterministic score over the three hyperparameters.</returns>
    private static double SpecifyParamsObjective(Trial t)
    {
        double baggingFraction = t.SuggestFloat("bagging_fraction", 0.4, 1.0 + 1e-12);
        long baggingFreq = t.SuggestInt("bagging_freq", 0, 7);
        long minChildSamples = t.SuggestInt("min_child_samples", 5, 100);
        return baggingFraction + 0.01 * baggingFreq + 0.001 * minChildSamples;
    }

    private const string PySpecifyParamsObjective =
        "def _obj(t):\n" +
        "    bf = t.suggest_float('bagging_fraction', 0.4, 1.0 + 1e-12)\n" +
        "    freq = t.suggest_int('bagging_freq', 0, 7)\n" +
        "    mcs = t.suggest_int('min_child_samples', 5, 100)\n" +
        "    return bf + 0.01 * freq + 0.001 * mcs\n";

    /// <summary>
    ///     Specify-Hyperparameters-Manually recipe: two <c>EnqueueTrial</c> calls (evaluated first) and two
    ///     <c>AddTrial(CreateTrial(...))</c> completed trials, followed by sampled optimization, must produce
    ///     an identical trial ledger (numbers, states, fixed vs sampled params, distributions) to Optuna.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SpecifyParams_EnqueueAndAddTrial_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomSampler(seed), studyName: "sp");
        cs.EnqueueTrial(new OrderedDict<string, object?>
        {
            ["bagging_fraction"] = 1.0, ["bagging_freq"] = 0, ["min_child_samples"] = 20,
        });
        cs.EnqueueTrial(new OrderedDict<string, object?>
        {
            ["bagging_fraction"] = 0.75, ["bagging_freq"] = 5, ["min_child_samples"] = 20,
        });

        var dists = new OrderedDict<string, BaseDistribution>
        {
            ["bagging_fraction"] = new FloatDistribution(0.4, 1.0 + 1e-12),
            ["bagging_freq"] = new IntDistribution(0, 7),
            ["min_child_samples"] = new IntDistribution(5, 100),
        };
        cs.AddTrial(FrozenTrial.CreateTrial(
            value: 0.94,
            @params: new OrderedDict<string, object?> { ["bagging_fraction"] = 1.0, ["bagging_freq"] = 0L, ["min_child_samples"] = 20L },
            distributions: dists));
        cs.AddTrial(FrozenTrial.CreateTrial(
            value: 0.95,
            @params: new OrderedDict<string, object?> { ["bagging_fraction"] = 0.75, ["bagging_freq"] = 5L, ["min_child_samples"] = 20L },
            distributions: dists));

        cs.Optimize(SpecifyParamsObjective, nTrials: 6);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='sp', direction='maximize', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.enqueue_trial({'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20})\n" +
            "study.enqueue_trial({'bagging_fraction': 0.75, 'bagging_freq': 5, 'min_child_samples': 20})\n" +
            "_dists = {'bagging_fraction': optuna.distributions.FloatDistribution(0.4, 1.0 + 1e-12), " +
            "'bagging_freq': optuna.distributions.IntDistribution(0, 7), " +
            "'min_child_samples': optuna.distributions.IntDistribution(5, 100)}\n" +
            "study.add_trial(optuna.trial.create_trial(params={'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20}, distributions=_dists, value=0.94))\n" +
            "study.add_trial(optuna.trial.create_trial(params={'bagging_fraction': 0.75, 'bagging_freq': 5, 'min_child_samples': 20}, distributions=_dists, value=0.95))\n" +
            PySpecifyParamsObjective +
            "study.optimize(_obj, n_trials=6)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ---------------------------------------------------------------- multi-objective.md

    /// <summary>
    ///     Multi-objective recipe: a two-objective study (minimize + maximize) must reproduce Optuna's full
    ///     trial ledger AND the derived Pareto <c>BestTrials</c> set exactly under a seeded RandomSampler.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void MultiObjective_DirectionsAndBestTrials_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            studyName: "mo",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize },
            sampler: new RandomSampler(seed));
        cs.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", 0, 5);
            double y = t.SuggestFloat("y", 0, 5);
            return new[] { x + y, x - y };
        }), nTrials: 30);
        var csState = StudySnapshot.Capture(cs);
        string csBest = string.Join(",", cs.BestTrials.Select(t => t.Number).OrderBy(n => n));

        PyExec(
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', 0, 5)\n" +
            "    y = t.suggest_float('y', 0, 5)\n" +
            "    return x + y, x - y\n" +
            $"study = optuna.create_study(study_name='mo', directions=['minimize', 'maximize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=30)\n" +
            "best = ','.join(str(n) for n in sorted(t.number for t in study.best_trials))\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
        csBest.Should().Be(PyStr("best"), "the derived Pareto best_trials set must match live Optuna");
    }

    // ---------------------------------------------------------------- ask-and-tell.md (define-by-run)

    /// <summary>Unwraps a categorical param value (which may be a <see cref="CategoricalChoice" />) to its scalar.</summary>
    /// <param name="value">The raw <c>Params</c> value.</param>
    /// <returns>The underlying scalar object.</returns>
    private static object? Unwrap(object? value) => value is CategoricalChoice c ? c.Value : value;

    /// <summary>
    ///     Drops the CMA-ES pickle-vs-JSON optimizer blob (<c>cma:optimizer:*</c>) from a snapshot before
    ///     parity: Optuna pickles its optimizer, OptunaSharp serializes its own JSON — a documented
    ///     <c>[Misaligned]</c> (cross-language mid-CMA resume is a non-goal). Every other field, including
    ///     the algorithmic <c>cma:generation</c> attribution, still compares bit-for-bit.
    /// </summary>
    /// <param name="s">Snapshot whose optimizer-state system attrs are removed in place.</param>
    private static void StripOptimizerAttrs(StudySnapshot s)
    {
        foreach (var t in s.Trials)
            foreach (var key in t.SystemAttrs.Keys.Where(k => k.Contains(":optimizer:")).ToList())
                t.SystemAttrs.Remove(key);
    }

    /// <summary>
    ///     Ask-and-Tell (define-by-run): an explicit <c>Ask</c>/<c>Tell</c> loop with a categorical and a
    ///     log-float parameter must reproduce Optuna's trial ledger exactly. The reported value uses only
    ///     bit-exact arithmetic so the recorded objective values also match.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void AskTell_DefineByRun_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomSampler(seed), studyName: "at1");
        for (int i = 0; i < 10; i++)
        {
            Trial trial = cs.Ask();
            double c = trial.SuggestFloat("C", 1e-7, 10.0, log: true);
            string solver = trial.SuggestCategorical("solver", new[] { "lbfgs", "saga" });
            cs.Tell(trial, c + (solver == "saga" ? 0.5 : 0.0));   // bit-exact: add only
        }
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(direction='maximize', study_name='at1', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "for _ in range(10):\n" +
            "    trial = study.ask()\n" +
            "    C = trial.suggest_float('C', 1e-7, 10.0, log=True)\n" +
            "    solver = trial.suggest_categorical('solver', ('lbfgs', 'saga'))\n" +
            "    study.tell(trial, C + (0.5 if solver == 'saga' else 0.0))\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ---------------------------------------------------------------- ask-and-tell.md (define-and-run)

    /// <summary>
    ///     Ask-and-Tell (define-and-run): pre-declared distributions passed to <c>Ask</c> are auto-sampled,
    ///     and the trial is finished from values read out of <c>Params</c> — the ledger must match Optuna.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void AskTell_DefineAndRun_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var distributions = new OrderedDict<string, BaseDistribution>
        {
            ["C"] = new FloatDistribution(1e-7, 10.0, log: true),
            ["solver"] = new CategoricalDistribution(new CategoricalChoice[] { "lbfgs", "saga" }),
        };

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomSampler(seed), studyName: "at2");
        for (int i = 0; i < 10; i++)
        {
            Trial trial = cs.Ask(distributions);
            double c = Convert.ToDouble(trial.Params["C"]);
            bool isSaga = (Unwrap(trial.Params["solver"]) as string) == "saga";
            cs.Tell(trial, c + (isSaga ? 0.5 : 0.0));
        }
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            "_dists = {'C': optuna.distributions.FloatDistribution(1e-7, 10.0, log=True), " +
            "'solver': optuna.distributions.CategoricalDistribution(('lbfgs', 'saga'))}\n" +
            $"study = optuna.create_study(direction='maximize', study_name='at2', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "for _ in range(10):\n" +
            "    trial = study.ask(_dists)\n" +
            "    C = trial.params['C']\n" +
            "    solver = trial.params['solver']\n" +
            "    study.tell(trial, C + (0.5 if solver == 'saga' else 0.0))\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ---------------------------------------------------------------- ask-and-tell.md (batch)

    /// <summary>
    ///     Ask-and-Tell (batch): asking a whole batch of trials before telling any, driven by a seeded
    ///     <see cref="CmaEsSampler" />, must reproduce Optuna's run — this exercises Tell-by-number and the
    ///     CMA-ES running-trials path together.
    /// </summary>
    /// <param name="seed">CmaEsSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void AskTell_BatchCmaEs_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        const int batchSize = 6;
        var cs = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: seed), studyName: "at3");
        for (int round = 0; round < 3; round++)
        {
            var numbers = new List<int>();
            var xs = new List<double>();
            var ys = new List<double>();
            for (int i = 0; i < batchSize; i++)
            {
                Trial trial = cs.Ask();
                numbers.Add(trial.Number);
                xs.Add(trial.SuggestFloat("x", -10, 10));
                ys.Add(trial.SuggestFloat("y", -10, 10));
            }
            for (int i = 0; i < batchSize; i++)
                cs.Tell(numbers[i], xs[i] * xs[i] + ys[i]);
        }
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='at3', sampler=optuna.samplers.CmaEsSampler(seed={seed}))\n" +
            "for _ in range(3):\n" +
            "    numbers = []; xs = []; ys = []\n" +
            "    for _ in range(6):\n" +
            "        trial = study.ask()\n" +
            "        numbers.append(trial.number)\n" +
            "        xs.append(trial.suggest_float('x', -10, 10))\n" +
            "        ys.append(trial.suggest_float('y', -10, 10))\n" +
            "    for n, x, y in zip(numbers, xs, ys):\n" +
            "        study.tell(n, x * x + y)\n");
        var pyState = PyStudyState("study");

        StripOptimizerAttrs(csState);
        StripOptimizerAttrs(pyState);
        StudySnapshot.AssertParity(pyState, csState);
    }

    // ---------------------------------------------------------------- reuse-best-trial.md

    /// <summary>Objective tuned by the study; a deterministic quadratic in a single log-float.</summary>
    /// <param name="t">The trial (live or frozen).</param>
    /// <returns>A maximized score peaking at C = 1.</returns>
    private static double ReuseObjective(BaseTrial t)
    {
        double c = t.SuggestFloat("C", 1e-7, 10.0, log: true);
        return -((c - 1.0) * (c - 1.0));
    }

    /// <summary>Second objective that replays the best trial's stored C and returns it as a "detailed metric".</summary>
    /// <param name="t">The frozen best trial.</param>
    /// <returns>The replayed C value.</returns>
    private static double DetailedObjective(BaseTrial t) => t.SuggestFloat("C", 1e-7, 10.0, log: true);

    /// <summary>
    ///     Re-use-the-best-trial recipe: after tuning, replaying <c>BestTrial</c> through a second objective
    ///     must return the best trial's stored parameter — bit-identical to running Optuna's detailed
    ///     objective on <c>study.best_trial</c> — and the study ledgers must match.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void ReuseBestTrial_Replay_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomSampler(seed), studyName: "rb");
        cs.Optimize(ReuseObjective, nTrials: 10);
        var csState = StudySnapshot.Capture(cs);
        double csDetailed = DetailedObjective(cs.BestTrial);

        PyExec(
            "def _obj(t):\n" +
            "    C = t.suggest_float('C', 1e-7, 10.0, log=True)\n" +
            "    return -((C - 1.0) * (C - 1.0))\n" +
            "def _detailed(t):\n" +
            "    return t.suggest_float('C', 1e-7, 10.0, log=True)\n" +
            $"study = optuna.create_study(direction='maximize', study_name='rb', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=10)\n" +
            "detailed = _detailed(study.best_trial)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
        csDetailed.Should().Be(PyFloat("detailed"), "replaying BestTrial must reproduce Optuna's best-trial parameter");
    }
}

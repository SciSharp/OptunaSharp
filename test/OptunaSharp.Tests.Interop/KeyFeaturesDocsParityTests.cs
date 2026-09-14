using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna deep-parity gate for the DocFX <b>Key Features</b> pages
///     (<c>docs/website-src/docs/key-features/</c>). Every code example on those pages is reproduced here
///     with a concrete, deterministic scenario and checked against real Optuna in the embedded interpreter:
///     study-mechanism examples capture with <see cref="StudySnapshot.Capture" /> and compare every trial
///     (number, state, values, external + internal params, distributions, attrs, intermediate values) via
///     <see cref="StudySnapshot.AssertParity" />; the visualization example compares the data-layer info
///     records field-by-field against Optuna's <c>_get_*_info</c> builders.
///
///     <para><b>Determinism.</b> The doc snippets are illustrative and use the default sampler; the tests
///     pin a seeded <see cref="RandomSampler" /> (bit-identical to NumPy's <c>RandomState</c> through
///     <c>LazyRandomState</c>/MT19937) so the run is reproducible, exactly as the "bit-for-bit under a fixed
///     seed" guarantee promises. Every objective uses multiplication and addition — never scalar
///     <c>**</c>, which differs from <c>x*x</c> in ~0.05% of cases — so the recorded values are bit-identical
///     on both runtimes. The examples whose <i>point</i> is a specific sampler/pruner use that component.</para>
///
///     <para><b>Scope.</b> The full per-plot visualization parity lives in
///     <see cref="VisualizationParityTests" />; this file pins the specific study and plots the Key Features
///     visualization page shows. Parallel execution (<c>n_jobs</c>) has no bit-parity in either runtime
///     because trial completion order is nondeterministic, so that example is anchored by a bit-exact
///     sequential run plus a structural check of the parallel run.</para>
/// </summary>
[TestClass]
public class KeyFeaturesDocsParityTests : OptunaInteropTestBase
{
    // ================================================================== first.md
    // "Lightweight, versatile, and platform-agnostic architecture"

    /// <summary>The quadratic <c>(x-2)²</c> objective of the first-page example (multiplication, not <c>**</c>).</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The squared error to minimize.</returns>
    private static double Quadratic(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        return (x - 2.0) * (x - 2.0);
    }

    /// <summary>Python source of <see cref="Quadratic" />, spelled identically so the stored values match bit-for-bit.</summary>
    private const string PyQuadratic =
        "def _q(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    return (x - 2.0) * (x - 2.0)\n";

    /// <summary>
    ///     first.md quadratic example: creating a study and optimizing the quadratic must reproduce Optuna's
    ///     full trial ledger, and the study result properties the page reads (<c>BestValue</c>,
    ///     <c>BestParams["x"]</c>, <c>Trials.Count</c>) must match live Optuna exactly.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void First_Quadratic_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "kf-first");
        cs.Optimize(Quadratic, nTrials: 40);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            PyQuadratic +
            $"study = optuna.create_study(study_name='kf-first', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_q, n_trials=40)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);

        // The properties the page prints must also match (redundant with the ledger, but the doc reads them).
        cs.BestValue.Should().Be(PyFloat("study.best_value"), "study.best_value in the example");
        Convert.ToDouble(cs.BestParams["x"]).Should().Be(PyFloat("study.best_params['x']"), "study.best_params['x']");
        cs.Trials.Count.Should().Be((int)PyLong("len(study.trials)"), "len(study.trials)");
    }

    /// <summary>
    ///     first.md "continue the optimization" example: calling <c>Optimize</c> a second time on the same
    ///     study appends trials (40 total) and the shared RandomSampler stream continues identically, so the
    ///     ledger stays bit-for-bit equal to Optuna running the same two calls.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void First_ContinueOptimization_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "kf-first-cont");
        cs.Optimize(Quadratic, nTrials: 20);
        cs.Optimize(Quadratic, nTrials: 20);   // continue — the page's point
        var csState = StudySnapshot.Capture(cs);
        cs.Trials.Count.Should().Be(40, "the second Optimize call appends 20 more trials");

        PyExec(
            PyQuadratic +
            $"study = optuna.create_study(study_name='kf-first-cont', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_q, n_trials=20)\n" +
            "study.optimize(_q, n_trials=20)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== search-space.md
    // "Pythonic search space"

    /// <summary>The seven-suggest objective of the search-space example — one of every distribution shape.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>A deterministic combination of every suggested parameter (addition order fixed for bit-parity).</returns>
    private static double SearchSpaceObjective(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "MomentumSGD", "Adam" });
        long numLayers = t.SuggestInt("num_layers", 1, 3);
        long numChannels = t.SuggestInt("num_channels", 32, 512, log: true);
        long numUnits = t.SuggestInt("num_units", 10, 100, step: 5);
        double dropoutRate = t.SuggestFloat("dropout_rate", 0.0, 1.0);
        double learningRate = t.SuggestFloat("learning_rate", 1e-5, 1e-2, log: true);
        double dropPathRate = t.SuggestFloat("drop_path_rate", 0.0, 1.0, step: 0.1);
        return dropoutRate + learningRate + dropPathRate + numLayers + numChannels + numUnits
               + (optimizer == "Adam" ? 1.0 : 0.0);
    }

    private const string PySearchSpaceObjective =
        "def _ss(t):\n" +
        "    optimizer = t.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])\n" +
        "    num_layers = t.suggest_int('num_layers', 1, 3)\n" +
        "    num_channels = t.suggest_int('num_channels', 32, 512, log=True)\n" +
        "    num_units = t.suggest_int('num_units', 10, 100, step=5)\n" +
        "    dropout_rate = t.suggest_float('dropout_rate', 0.0, 1.0)\n" +
        "    learning_rate = t.suggest_float('learning_rate', 1e-5, 1e-2, log=True)\n" +
        "    drop_path_rate = t.suggest_float('drop_path_rate', 0.0, 1.0, step=0.1)\n" +
        "    return dropout_rate + learning_rate + drop_path_rate + num_layers + num_channels + num_units \\\n" +
        "        + (1.0 if optimizer == 'Adam' else 0.0)\n";

    /// <summary>
    ///     search-space.md example: exercising every <c>Suggest*</c> shape (categorical, int, int-log,
    ///     int-step, float, float-log, float-step) must reproduce Optuna's trials AND, critically, the exact
    ///     distribution JSON and internal representation of each parameter — the deep check that step/log
    ///     discretization and internal reprs are byte-identical.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SearchSpace_AllDistributionShapes_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "kf-ss");
        cs.Optimize(SearchSpaceObjective, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            PySearchSpaceObjective +
            $"study = optuna.create_study(study_name='kf-ss', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_ss, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    /// <summary>Branching objective from the search-space example — the active parameter depends on a categorical.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The single log-parameter chosen by the branch.</returns>
    private static double BranchesObjective(Trial t)
    {
        string classifier = t.SuggestCategorical("classifier", new[] { "SVC", "RandomForest" });
        if (classifier == "SVC")
            return t.SuggestFloat("svc_c", 1e-10, 1e10, log: true);
        return t.SuggestInt("rf_max_depth", 2, 32, log: true);
    }

    private const string PyBranchesObjective =
        "def _br(t):\n" +
        "    classifier = t.suggest_categorical('classifier', ['SVC', 'RandomForest'])\n" +
        "    if classifier == 'SVC':\n" +
        "        return t.suggest_float('svc_c', 1e-10, 1e10, log=True)\n" +
        "    return t.suggest_int('rf_max_depth', 2, 32, log=True)\n";

    /// <summary>
    ///     search-space.md branches example: a conditional search space (only one of <c>svc_c</c> /
    ///     <c>rf_max_depth</c> is defined per trial) must reproduce Optuna's per-trial parameter sets and
    ///     distributions exactly.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SearchSpace_Branches_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "kf-br");
        cs.Optimize(BranchesObjective, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            PyBranchesObjective +
            $"study = optuna.create_study(study_name='kf-br', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_br, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    /// <summary>Loop objective from the search-space example — a suggested layer count drives per-layer params.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The sum of the per-layer unit counts (addition of exact integers → bit-parity).</returns>
    private static double LoopsObjective(Trial t)
    {
        long nLayers = t.SuggestInt("n_layers", 1, 3);
        double sum = 0.0;
        for (int i = 0; i < nLayers; i++)
            sum += t.SuggestInt($"n_units_l{i}", 4, 128, log: true);
        return sum;
    }

    private const string PyLoopsObjective =
        "def _lp(t):\n" +
        "    n_layers = t.suggest_int('n_layers', 1, 3)\n" +
        "    s = 0.0\n" +
        "    for i in range(n_layers):\n" +
        "        s += t.suggest_int(f'n_units_l{i}', 4, 128, log=True)\n" +
        "    return s\n";

    /// <summary>
    ///     search-space.md loops example: a dynamic search space whose parameter <i>count</i> varies per trial
    ///     (<c>n_units_l0</c>, <c>n_units_l1</c>, …) must reproduce Optuna's trials exactly, proving the
    ///     define-by-run loop is byte-identical.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SearchSpace_Loops_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "kf-lp");
        cs.Optimize(LoopsObjective, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            PyLoopsObjective +
            $"study = optuna.create_study(study_name='kf-lp', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_lp, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== efficient-algorithms.md
    // "Efficient optimization algorithms"

    /// <summary>
    ///     efficient-algorithms.md "switching samplers" example: the default sampler and each explicitly
    ///     constructed sampler must report the same class name OptunaSharp shows via
    ///     <c>study.Sampler.GetType().Name</c> as Optuna shows via
    ///     <c>type(study.sampler).__name__</c> — including the shared <c>TPESampler</c> default.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void EfficientAlgorithms_SwitchingSamplers_Parity()
    {
        PythonSession.EnsureOrInconclusive();

        Optuna.CreateStudy().Sampler.GetType().Name
            .Should().Be(PyStr("type(optuna.create_study().sampler).__name__"), "default sampler class name");
        Optuna.CreateStudy(sampler: new RandomSampler()).Sampler.GetType().Name
            .Should().Be(PyStr("type(optuna.create_study(sampler=optuna.samplers.RandomSampler()).sampler).__name__"));
        Optuna.CreateStudy(sampler: new CmaEsSampler()).Sampler.GetType().Name
            .Should().Be(PyStr("type(optuna.create_study(sampler=optuna.samplers.CmaEsSampler()).sampler).__name__"));

        Optuna.CreateStudy().Sampler.GetType().Name.Should().Be("TPESampler", "the documented default");
    }

    /// <summary>Iterative objective from the pruning example — reports a deterministic per-step value and prunes.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The final reported error, unless the trial is pruned first.</returns>
    /// <exception cref="TrialPruned">Raised when the pruner stops the trial early (recorded as Pruned).</exception>
    private static double PrunerObjective(Trial t)
    {
        double alpha = t.SuggestFloat("alpha", 1e-5, 1e-1, log: true);
        double intermediate = 0.0;
        for (int step = 0; step < 20; step++)
        {
            intermediate = alpha * (20 - step);   // deterministic; larger alpha ⇒ worse ⇒ pruned
            t.Report(intermediate, step);
            if (t.ShouldPrune())
                throw new TrialPruned();
        }
        return intermediate;
    }

    private const string PyPrunerObjective =
        "def _pr(t):\n" +
        "    alpha = t.suggest_float('alpha', 1e-5, 1e-1, log=True)\n" +
        "    intermediate = 0.0\n" +
        "    for step in range(20):\n" +
        "        intermediate = alpha * (20 - step)\n" +
        "        t.report(intermediate, step)\n" +
        "        if t.should_prune():\n" +
        "            raise optuna.TrialPruned()\n" +
        "    return intermediate\n";

    /// <summary>
    ///     efficient-algorithms.md "activating a pruner" example: <c>Report</c>/<c>ShouldPrune</c> with a
    ///     <see cref="MedianPruner" /> must produce the identical prune/complete decisions, intermediate values
    ///     and final values as live Optuna — the deep proof that the pruning mechanism matches, including which
    ///     trials are pruned and at which step.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void EfficientAlgorithms_MedianPrunerActivation_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            sampler: new RandomSampler(seed), pruner: new MedianPruner(), studyName: "kf-prune");
        cs.Optimize(PrunerObjective, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);
        cs.GetTrials().Any(t => t.State == TrialState.Pruned)
            .Should().BeTrue("the median rule should prune at least one trial in 30");

        PyExec(
            PyPrunerObjective +
            $"study = optuna.create_study(study_name='kf-prune', sampler=optuna.samplers.RandomSampler(seed={seed}), " +
            "pruner=optuna.pruners.MedianPruner())\n" +
            "study.optimize(_pr, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== parallelization.md
    // "Easy parallelization"

    /// <summary>
    ///     parallelization.md multi-process example: creating a study on a shared <see cref="JournalStorage" />,
    ///     then re-opening it with <c>loadIfExists</c> and optimizing more, must reproduce Optuna's full ledger.
    ///     A resumed run uses a fresh (re-seeded) sampler on both sides — exactly the page's "storage doesn't
    ///     persist sampler state" note — and the 20-trial result matches bit-for-bit.
    /// </summary>
    /// <param name="seed">Seed for the first phase; the resume phase uses <paramref name="seed" /> + 1 on both sides.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Parallelization_MultiProcessResume_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        string csPath = TempJournalPath();
        string pyPath = TempJournalPath();
        try
        {
            // Phase 1, then resume with load_if_exists and a fresh sampler (the doc's multi-process pattern).
            var s1 = Optuna.CreateStudy(
                studyName: "kf-par", storage: NewJournal(csPath), sampler: new RandomSampler(seed));
            s1.Optimize(Quadratic, nTrials: 10);
            var s2 = Optuna.CreateStudy(
                studyName: "kf-par", storage: NewJournal(csPath), sampler: new RandomSampler(seed + 1),
                loadIfExists: true);
            s2.Optimize(Quadratic, nTrials: 10);
            var csState = StudySnapshot.Capture(s2);
            csState.Trials.Count.Should().Be(20, "10 initial + 10 resumed trials");

            PyExec(
                PyQuadratic +
                $"st1 = {PyJournalStorageExpr(pyPath)}\n" +
                $"study = optuna.create_study(study_name='kf-par', storage=st1, sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
                "study.optimize(_q, n_trials=10)\n" +
                $"st2 = {PyJournalStorageExpr(pyPath)}\n" +
                $"study = optuna.create_study(study_name='kf-par', storage=st2, sampler=optuna.samplers.RandomSampler(seed={seed + 1}), load_if_exists=True)\n" +
                "study.optimize(_q, n_trials=10)\n");

            StudySnapshot.AssertParity(PyStudyState("study"), csState);
        }
        finally
        {
            DeleteJournal(csPath);
            DeleteJournal(pyPath);
        }
    }

    /// <summary>
    ///     parallelization.md multi-thread example: <c>Optimize(..., nJobs: 4)</c> over a shared
    ///     <see cref="JournalStorage" /> must complete every requested trial. Parallel completion order is
    ///     nondeterministic (in Optuna too), so this is a structural guarantee — 20 complete trials with a
    ///     finite best — anchored by a bit-exact <c>nJobs: 1</c> run against live Optuna in the same test.
    /// </summary>
    /// <param name="seed">RandomSampler seed; the sequential anchor shares it with Optuna.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Parallelization_NJobs_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Bit-exact anchor: nJobs=1 over a shared journal reproduces Optuna's sequential run exactly.
        string seqPath = TempJournalPath();
        try
        {
            var seq = Optuna.CreateStudy(
                studyName: "kf-seq", storage: NewJournal(seqPath), sampler: new RandomSampler(seed));
            seq.Optimize(Quadratic, nTrials: 20, nJobs: 1);
            var seqState = StudySnapshot.Capture(seq);

            PyExec(
                PyQuadratic +
                $"study = optuna.create_study(study_name='kf-seq', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
                "study.optimize(_q, n_trials=20)\n");
            StudySnapshot.AssertParity(PyStudyState("study"), seqState);
        }
        finally
        {
            DeleteJournal(seqPath);
        }

        // Structural: nJobs=4 still produces the full, all-complete study (no bit-parity — parallel order varies).
        string parPath = TempJournalPath();
        try
        {
            var par = Optuna.CreateStudy(
                studyName: "kf-par4", storage: NewJournal(parPath), sampler: new RandomSampler(seed));
            par.Optimize(Quadratic, nTrials: 20, nJobs: 4);

            par.Trials.Count.Should().Be(20, "every requested trial runs under n_jobs=4");
            par.GetTrials().All(t => t.State == TrialState.Complete).Should().BeTrue("all parallel trials complete");
            double.IsFinite(par.BestValue).Should().BeTrue("a finite best is found under parallel execution");
        }
        finally
        {
            DeleteJournal(parPath);
        }
    }

    // ================================================================== visualization.md
    // "Quick visualization"

    /// <summary>Iterative multi-parameter objective for the visualization study — reports a per-epoch accuracy.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The final accuracy (maximized), unless the trial is pruned first.</returns>
    /// <exception cref="TrialPruned">Raised when the median rule stops the trial early.</exception>
    private static double VizObjective(Trial t)
    {
        long nLayers = t.SuggestInt("n_layers", 1, 3);
        long nUnits = t.SuggestInt("n_units", 64, 512, log: true);
        double lr = t.SuggestFloat("lr", 1e-5, 1e-1, log: true);
        double accuracy = 0.0;
        for (int epoch = 0; epoch < 10; epoch++)
        {
            accuracy = lr * 5.0 + nLayers * 0.01 + nUnits * 0.0001 + epoch * 0.02;
            t.Report(accuracy, epoch);
            if (t.ShouldPrune())
                throw new TrialPruned();
        }
        return accuracy;
    }

    private const string PyVizObjective =
        "def _viz(t):\n" +
        "    n_layers = t.suggest_int('n_layers', 1, 3)\n" +
        "    n_units = t.suggest_int('n_units', 64, 512, log=True)\n" +
        "    lr = t.suggest_float('lr', 1e-5, 1e-1, log=True)\n" +
        "    accuracy = 0.0\n" +
        "    for epoch in range(10):\n" +
        "        accuracy = lr * 5.0 + n_layers * 0.01 + n_units * 0.0001 + epoch * 0.02\n" +
        "        t.report(accuracy, epoch)\n" +
        "        if t.should_prune():\n" +
        "            raise optuna.TrialPruned()\n" +
        "    return accuracy\n";

    /// <summary>Importance values inherit the forest/erf few-ULP tolerance; names, order and labels are exact.</summary>
    private const double ImportanceTolerance = 1e-10;

    /// <summary>
    ///     visualization.md example: for the page's exact study (maximize + seeded sampler + MedianPruner over an
    ///     iterative objective), the data-layer info records behind the plot calls must match live Optuna —
    ///     <c>PlotOptimizationHistory</c> (<see cref="OptimizationHistory" />) and
    ///     <c>PlotIntermediateValues</c> (<see cref="IntermediateValues" />) bit-for-bit, and
    ///     <c>PlotParamImportances</c> (<see cref="ParamImportances" />) to the forest/erf tolerance. The full
    ///     per-plot gallery (contour/slice/parallel-coordinate/edf/rank/timeline) is covered by
    ///     <see cref="VisualizationParityTests" />.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes (pinned so the study is bit-identical).</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void Visualization_HistoryIntermediateImportances_Parity(int seed)
    {
        // Bit-exact PyFloat comparisons below — host-pinned to x64 (arm64 libm drifts ~1 ULP from NumPy).
        PythonSession.EnsureBitExactHostOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomSampler(seed), pruner: new MedianPruner());
        cs.Optimize(VizObjective, nTrials: 40);

        var history = OptimizationHistory.GetInfoList(cs);
        var intermediate = IntermediateValues.GetPlotInfo(cs);
        var importances = ParamImportances.GetInfo(cs);

        PyExec(
            PyVizObjective +
            $"study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed={seed}), " +
            "pruner=optuna.pruners.MedianPruner())\n" +
            "study.optimize(_viz, n_trials=40)\n" +
            "from optuna.visualization._optimization_history import _get_optimization_history_info_list\n" +
            "from optuna.visualization._intermediate_values import _get_intermediate_plot_info\n" +
            "from optuna.visualization._param_importances import _get_importances_info\n" +
            "hist = _get_optimization_history_info_list(study, None, 'Objective Value', False)\n" +
            "inter = _get_intermediate_plot_info(study)\n" +
            "imp = _get_importances_info(study, None, None, None, 'Objective Value')\n");

        // --- optimization history (pure numpy over the trials → bit-exact) ---
        PyLong("len(hist)").Should().Be(history.Count);
        var h = history[0];
        PyLong("len(hist[0].trial_numbers)").Should().Be(h.TrialNumbers.Count);
        for (int i = 0; i < h.TrialNumbers.Count; i++)
            PyLong($"hist[0].trial_numbers[{i}]").Should().Be(h.TrialNumbers[i], $"history trial_numbers[{i}]");
        PyStr("hist[0].values_info.label_name").Should().Be(h.ValuesInfo.LabelName);
        PyLong("len(hist[0].values_info.values)").Should().Be(h.ValuesInfo.Values.Count);
        for (int i = 0; i < h.ValuesInfo.Values.Count; i++)
        {
            PyFloat($"hist[0].values_info.values[{i}]").Should().Be(h.ValuesInfo.Values[i], $"history values[{i}]");
            PyLong($"int(hist[0].values_info.states[{i}].value)").Should().Be((int)h.ValuesInfo.States[i], $"history states[{i}]");
        }

        // --- intermediate values (learning curves; bit-exact) ---
        PyLong("len(inter.trial_infos)").Should().Be(intermediate.TrialInfos.Count);
        for (int k = 0; k < intermediate.TrialInfos.Count; k++)
        {
            var ti = intermediate.TrialInfos[k];
            PyLong($"inter.trial_infos[{k}].trial_number").Should().Be(ti.TrialNumber, $"intermediate trial_number[{k}]");
            int m = (int)PyLong($"len(inter.trial_infos[{k}].sorted_intermediate_values)");
            m.Should().Be(ti.SortedIntermediateValues.Count, $"intermediate step count[{k}]");
            for (int j = 0; j < m; j++)
            {
                PyLong($"inter.trial_infos[{k}].sorted_intermediate_values[{j}][0]")
                    .Should().Be(ti.SortedIntermediateValues[j].Step, $"intermediate[{k}] step[{j}]");
                PyFloat($"inter.trial_infos[{k}].sorted_intermediate_values[{j}][1]")
                    .Should().Be(ti.SortedIntermediateValues[j].Value, $"intermediate[{k}] value[{j}]");
            }
        }

        // --- param importances (fANOVA over a forest → few-ULP erf tolerance; names/labels exact) ---
        PyStr("imp.target_name").Should().Be(importances.TargetName);
        int np = (int)PyLong("len(imp.param_names)");
        np.Should().Be(importances.ParamNames.Count);
        for (int i = 0; i < np; i++)
        {
            PyStr($"imp.param_names[{i}]").Should().Be(importances.ParamNames[i], $"importance param_names[{i}]");
            PyFloat($"imp.importance_values[{i}]")
                .Should().BeApproximately(importances.ImportanceValues[i], ImportanceTolerance, $"importance_values[{i}]");
        }
    }

    // ================================================================== journal-file helpers

    /// <summary>A unique temp path for a journal file, forward-slashed for the Python side.</summary>
    /// <returns>An absolute journal-file path that does not yet exist.</returns>
    private static string TempJournalPath()
        => Path.Combine(Path.GetTempPath(), $"oss-kf-{Guid.NewGuid():N}.log").Replace('\\', '/');

    /// <summary>Constructs a Core journal storage over <paramref name="path" /> with the Windows-safe open lock.</summary>
    /// <param name="path">Journal file path.</param>
    /// <returns>A fresh <see cref="JournalStorage" /> instance over that file.</returns>
    private static JournalStorage NewJournal(string path)
        => new(new JournalFileBackend(path, lockObj: new JournalFileOpenLock(path)));

    /// <summary>Best-effort removal of a journal file and any lock siblings it left behind.</summary>
    /// <param name="path">Journal file path passed to <see cref="NewJournal" />.</param>
    private static void DeleteJournal(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (dir is null)
            return;
        foreach (var f in Directory.GetFiles(dir, Path.GetFileName(path) + "*"))
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
                // A lock sibling may still be held briefly; leaving a temp file is harmless.
            }
        }
    }
}

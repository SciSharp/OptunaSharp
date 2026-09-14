using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna deep-parity gate for the DocFX <b>FAQ</b> page (<c>docs/website-src/docs/faq.md</c>).
///     Each answer whose example is a concrete, deterministic program is reproduced here and checked against
///     real Optuna in the embedded interpreter: study-mechanism examples capture with
///     <see cref="StudySnapshot.Capture" /> and compare every trial (number, state, values, external + internal
///     params, distributions, user/system attrs — constraints included, since <c>SetConstraint</c> writes them
///     to system attrs — and intermediate values) via <see cref="StudySnapshot.AssertParity" />; the study-free
///     examples (<see cref="FixedTrial" />, logging verbosity) compare their scalar results directly.
///
///     <para><b>Determinism.</b> The FAQ snippets are illustrative and mostly use the default sampler; the tests
///     pin a seeded <see cref="RandomSampler" /> (bit-identical to NumPy's <c>RandomState</c> through
///     <c>LazyRandomState</c>/MT19937) so the run is reproducible — exactly the "bit-for-bit under a fixed seed"
///     guarantee the reproducibility answer promises. Every objective uses multiplication/addition and
///     <c>Math.Sqrt</c>/<c>Math.Log</c> (never scalar <c>**</c>, which diverges from <c>x*x</c> in ~0.05% of
///     cases), and the Python side is spelled identically, so recorded values are bit-identical on both runtimes.
///     The example whose <i>point</i> is a specific sampler (constrained NSGA-II) uses that sampler.</para>
///
///     <para><b>Out of scope</b> (not deterministically parity-testable here, covered elsewhere, or a mere
///     pointer): "use with X"/GPU env vars/starting-points (prose or covered by
///     <see cref="RecipeParityTests" />), heartbeat monitoring (experimental + timing-dependent), SQLite/journal
///     parallel storage (covered by the storage parity files), and multi-thread <c>nJobs</c> (nondeterministic
///     completion order — anchored in <see cref="KeyFeaturesDocsParityTests" />).</para>
/// </summary>
[TestClass]
public class FaqDocsParityTests : OptunaInteropTestBase
{
    // ================================================================== shared quadratic
    // The FAQ's recurring "(x - 2)²" objective, spelled with multiplication so both runtimes store
    // bit-identical values. Reused by the own-arguments, reproducibility, OOM, and save/resume answers.

    /// <summary>The quadratic <c>(x-2)²</c> over <c>x ∈ [-10, 10]</c> (multiplication, never <c>**</c>).</summary>
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

    // ================================================================== own arguments
    // "How to define objective functions that have own arguments?"

    /// <summary>Callable-class objective holding its bounds as fields — the FAQ's first "own arguments" form.</summary>
    private sealed class OwnArgsObjective
    {
        private readonly double _minX;
        private readonly double _maxX;

        /// <param name="minX">Inclusive lower bound passed to <c>SuggestFloat</c>.</param>
        /// <param name="maxX">Inclusive upper bound passed to <c>SuggestFloat</c>.</param>
        public OwnArgsObjective(double minX, double maxX)
        {
            _minX = minX;
            _maxX = maxX;
        }

        /// <summary>Objective body; signature matches <c>Func&lt;Trial, double&gt;</c>.</summary>
        /// <param name="trial">The trial being evaluated.</param>
        /// <returns>The squared error over the class-held bounds.</returns>
        public double Invoke(Trial trial)
        {
            double x = trial.SuggestFloat("x", _minX, _maxX);
            return (x - 2.0) * (x - 2.0);
        }
    }

    /// <summary>The FAQ's second "own arguments" form — a free function closed over by a lambda at the call site.</summary>
    /// <param name="trial">The trial being evaluated.</param>
    /// <param name="minX">Inclusive lower bound.</param>
    /// <param name="maxX">Inclusive upper bound.</param>
    /// <returns>The squared error over the captured bounds.</returns>
    private static double OwnArgsLambdaObjective(Trial trial, double minX, double maxX)
    {
        double x = trial.SuggestFloat("x", minX, maxX);
        return (x - 2.0) * (x - 2.0);
    }

    /// <summary>
    ///     "Own arguments" answer: the callable-class form must reproduce Optuna's callable-class run bit-for-bit,
    ///     and the lambda-closure form must produce the identical study to the class form — proving both documented
    ///     ways of passing extra arguments behave the same.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes and by both C# forms.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void OwnArguments_ClassAndLambda_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csClass = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-args");
        csClass.Optimize(new OwnArgsObjective(-100, 100).Invoke, nTrials: 40);
        var classState = StudySnapshot.Capture(csClass);

        var csLambda = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-args");
        csLambda.Optimize(t => OwnArgsLambdaObjective(t, -100, 100), nTrials: 40);
        // The two documented forms must yield an identical study under the same seed.
        StudySnapshot.AssertParity(classState, StudySnapshot.Capture(csLambda));

        PyExec(
            "class _Obj:\n" +
            "    def __init__(self, min_x, max_x):\n" +
            "        self.min_x = min_x\n" +
            "        self.max_x = max_x\n" +
            "    def __call__(self, trial):\n" +
            "        x = trial.suggest_float('x', self.min_x, self.max_x)\n" +
            "        return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(study_name='faq-args', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_Obj(-100, 100), n_trials=40)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), classState);
    }

    // ================================================================== reproducibility
    // "How can I obtain reproducible optimization results?"

    /// <summary>
    ///     Reproducibility answer: a seeded sampler is deterministic (two C# runs are bit-identical) AND matches
    ///     live Optuna bit-for-bit under the same seed — the RandomSampler pin makes the cross-runtime guarantee
    ///     assertable, standing in for the FAQ's illustrative seeded TPESampler.
    /// </summary>
    /// <param name="seed">Sampler seed shared by both C# runs and Optuna.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(10)]
    [DataRow(42)]
    public void Reproducibility_SameSeed_DeterministicAndMatchesOptuna(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var run1 = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-repro");
        run1.Optimize(Quadratic, nTrials: 30);
        var run2 = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-repro");
        run2.Optimize(Quadratic, nTrials: 30);

        // Same seed ⇒ bit-identical studies (the reproducibility claim).
        StudySnapshot.AssertParity(StudySnapshot.Capture(run1), StudySnapshot.Capture(run2));

        PyExec(
            PyQuadratic +
            $"study = optuna.create_study(study_name='faq-repro', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_q, n_trials=30)\n");

        // …and it matches Python bit-for-bit (the cross-runtime guarantee).
        StudySnapshot.AssertParity(PyStudyState("study"), StudySnapshot.Capture(run1));
    }

    /// <summary>
    ///     Reproducibility answer, verbatim FAQ snippet: <c>new TPESampler(seed: 10)</c> is bit-reproducible —
    ///     two seeded C# runs produce the identical study. (TPE's cross-runtime parity uses erf tolerance and is
    ///     covered by <see cref="TpeSamplerParityTests" />; here we assert only same-seed determinism, which is
    ///     bit-exact within one runtime.)
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Reproducibility_SeededTpe_IsBitReproducible()
    {
        PythonSession.EnsureOrInconclusive();

        var a = Optuna.CreateStudy(sampler: new TPESampler(seed: 10), studyName: "faq-tpe");
        a.Optimize(Quadratic, nTrials: 25);
        var b = Optuna.CreateStudy(sampler: new TPESampler(seed: 10), studyName: "faq-tpe");
        b.Optimize(Quadratic, nTrials: 25);

        StudySnapshot.AssertParity(StudySnapshot.Capture(a), StudySnapshot.Capture(b));
    }

    // ================================================================== suppress log messages
    // "How to suppress log messages of OptunaSharp?"

    /// <summary>
    ///     Logging answer: <c>OptunaLogging.SetVerbosity(LogLevel.Warning)</c> must map to the same numeric level
    ///     Optuna's <c>set_verbosity(WARNING)</c> uses (30) — the shared level scale the doc relies on. The global
    ///     verbosity is saved and restored so this test does not perturb sibling tests.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SuppressLogs_WarningVerbosity_MatchesOptunaLevel()
    {
        PythonSession.EnsureOrInconclusive();

        int original = OptunaLogging.GetVerbosity();   // LogLevel members are int constants, not an enum
        try
        {
            OptunaLogging.SetVerbosity(LogLevel.Warning);
            ((long)OptunaLogging.GetVerbosity())
                .Should().Be(PyLong("optuna.logging.get_verbosity()"), "WARNING must be the same numeric level");
            OptunaLogging.GetVerbosity().Should().Be(30, "Optuna's WARNING level is 30");
        }
        finally
        {
            OptunaLogging.SetVerbosity(original);   // restore global state for other tests
        }
    }

    // ================================================================== exceptions & the catch argument
    // "How are exceptions from trials handled?"

    /// <summary>
    ///     Exceptions answer: an objective that throws for some trials must fail exactly those trials
    ///     (state = FAIL, values = null, the pre-throw params retained) and — with the offending type passed to
    ///     <c>@catch</c> — the study continues, reproducing Optuna's <c>catch=(...)</c> run bit-for-bit.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Exceptions_CatchArgument_ContinuesAndFailsParity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-catch");
        cs.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            if (x > 5.0)
                throw new ArithmeticException("A sample error in objective.");   // becomes a FAIL trial
            return (x - 2.0) * (x - 2.0);
        }, nTrials: 30, @catch: new[] { typeof(ArithmeticException) });
        var csState = StudySnapshot.Capture(cs);
        cs.GetTrials().Any(t => t.State == TrialState.Fail)
            .Should().BeTrue("some trials sample x > 5 and are caught as failures");

        PyExec(
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    if x > 5.0:\n" +
            "        raise ValueError('A sample error in objective.')\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(study_name='faq-catch', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=30, catch=(ValueError,))\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== NaN handling
    // "How are NaNs returned by trials handled?"

    /// <summary>
    ///     NaN answer: a trial that returns <c>double.NaN</c> is failed (state = FAIL, values = null) but does not
    ///     abort the study — reproducing Optuna's NaN-return handling bit-for-bit, with the completed trials still
    ///     carrying their finite values.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Nans_ReturnedNaN_AreFailedButStudyContinues_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-nan");
        cs.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return x > 5.0 ? double.NaN : (x - 2.0) * (x - 2.0);   // NaN ⇒ FAIL, does not abort
        }, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);
        cs.GetTrials().Count(t => t.State == TrialState.Fail)
            .Should().BeGreaterThan(0, "trials returning NaN are recorded as failures");
        cs.GetTrials().Count(t => t.State == TrialState.Complete)
            .Should().BeGreaterThan(0, "the study keeps running after NaN failures");

        PyExec(
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return float('nan') if x > 5.0 else (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(study_name='faq-nan', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== avoiding OOM
    // "How do I avoid running out of memory (OOM) when optimizing studies?"

    /// <summary>
    ///     OOM answer: <c>gcAfterTrial: true</c> is a pure side-channel — it must not change the study, so the
    ///     gc-enabled run equals both a plain C# run and Optuna's <c>gc_after_trial=True</c> run bit-for-bit.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Oom_GcAfterTrial_DoesNotChangeResults_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var gc = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-gc");
        gc.Optimize(Quadratic, nTrials: 20, gcAfterTrial: true);
        var gcState = StudySnapshot.Capture(gc);

        var plain = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-gc");
        plain.Optimize(Quadratic, nTrials: 20);   // gcAfterTrial defaults to false
        StudySnapshot.AssertParity(StudySnapshot.Capture(plain), gcState);

        PyExec(
            PyQuadratic +
            $"study = optuna.create_study(study_name='faq-gc', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_q, n_trials=20, gc_after_trial=True)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), gcState);
    }

    // ================================================================== log only on best update
    // "How can I output a log only when the best value is updated?"

    /// <summary>
    ///     Best-update-callback answer: the callback stores the last best value in <c>study.user_attrs</c>
    ///     (<c>previous_best_value</c>); after the run that attribute holds the final best, and the whole study —
    ///     attributes included — matches Optuna running the identical callback bit-for-bit.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void LogOnBestUpdate_Callback_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-bestlog");

        // The doc's callback, minus the Console.WriteLine (printing has no bearing on study state/parity).
        void LoggingCallback(Study study, FrozenTrial frozenTrial)
        {
            object? previousBest = study.UserAttrs.GetValueOrDefault("previous_best_value");
            if (!Equals(previousBest, study.BestValue))
                study.SetUserAttr("previous_best_value", study.BestValue);
        }

        cs.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            return x * x;
        }, nTrials: 30, callbacks: new Action<Study, FrozenTrial>[] { LoggingCallback });
        var csState = StudySnapshot.Capture(cs);
        cs.UserAttrs.Should().ContainKey("previous_best_value", "the callback records the running best");

        PyExec(
            "def _cb(study, frozen_trial):\n" +
            "    prev = study.user_attrs.get('previous_best_value', None)\n" +
            "    if prev != study.best_value:\n" +
            "        study.set_user_attr('previous_best_value', study.best_value)\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', 0, 1)\n" +
            "    return x * x\n" +
            $"study = optuna.create_study(study_name='faq-bestlog', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=30, callbacks=[_cb])\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== Dirichlet proportions
    // "How do I suggest variables that represent proportions (a Dirichlet distribution)?"

    /// <summary>Tolerance for the Dirichlet proportions. The ≤1-ULP cross-runtime drift is NOT from
    ///     <c>Math.Log</c> (verified bit-exact vs ucrt <c>math.log</c>/<c>np.log</c>) but from the normalization
    ///     denominator: <b>CPython 3.12's built-in <c>sum()</c> uses Neumaier compensated summation</b>, whereas
    ///     C#'s <c>Enumerable.Sum</c> is uncompensated, so <c>x_i / sum</c> can differ in the last bit. An absolute
    ///     1e-9 dwarfs the observed ~1e-16 gap.</summary>
    private const double DirichletTolerance = 1e-9;

    /// <summary>
    ///     Dirichlet answer: the <c>-log(u)</c>/normalize trick stores each proportion as a per-trial user
    ///     attribute. The sampled draws <c>x_i</c> are sampler-controlled and compared <b>bit-exact</b> (the
    ///     "bit-for-bit suggested parameters" guarantee), while the user-computed proportions <c>p_i</c> are
    ///     compared to <see cref="DirichletTolerance" /> — the doc's <c>xs.Sum()</c> normalization differs from
    ///     CPython 3.12's compensated <c>sum()</c> by at most one ULP, so bit-parity is neither claimed nor
    ///     required for this user-space arithmetic. Each trial's proportions are also asserted to sum to one
    ///     (the Dirichlet property).
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Dirichlet_Proportions_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        const int n = 5;
        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-dir");
        cs.Optimize(t =>
        {
            var xs = new double[n];
            for (int i = 0; i < n; i++)
                xs[i] = -Math.Log(t.SuggestFloat($"x_{i}", 0, 1));
            double sum = xs.Sum();
            for (int i = 0; i < n; i++)
                t.SetUserAttr($"p_{i}", xs[i] / sum);
            return 0.0;
        }, nTrials: 60);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            "def _obj(t):\n" +
            "    n = 5\n" +
            "    xs = [-math.log(t.suggest_float(f'x_{i}', 0, 1)) for i in range(n)]\n" +
            "    s = sum(xs)\n" +
            "    for i in range(n):\n" +
            "        t.set_user_attr(f'p_{i}', xs[i] / s)\n" +
            "    return 0.0\n" +
            $"study = optuna.create_study(study_name='faq-dir', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=60)\n");
        var pyState = PyStudyState("study");

        pyState.Trials.Count.Should().Be(csState.Trials.Count, "trial count");
        for (int k = 0; k < csState.Trials.Count; k++)
        {
            TrialSnapshot ct = csState.Trials[k];
            TrialSnapshot pt = pyState.Trials[k];
            ct.Number.Should().Be(pt.Number);

            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                // The sampled draw x_i is sampler-controlled ⇒ bit-exact across runtimes.
                ((double)ct.Params[$"x_{i}"]!).Should().Be((double)pt.Params[$"x_{i}"]!,
                    $"trial[{k}] x_{i} is a sampler-controlled draw");

                // The proportion p_i is user-space Math.Log math ⇒ agree to floating tolerance.
                double csP = Convert.ToDouble(ct.UserAttrs[$"p_{i}"]);
                double pyP = Convert.ToDouble(pt.UserAttrs[$"p_{i}"]);
                csP.Should().BeApproximately(pyP, DirichletTolerance, $"trial[{k}] proportion p_{i}");
                sum += csP;
            }

            sum.Should().BeApproximately(1.0, DirichletTolerance, $"trial[{k}] proportions sum to 1");
        }
    }

    // ================================================================== constraints
    // "How can I optimize a model with some constraints?"

    /// <summary>
    ///     Constraints answer: the Binh–Korn constrained multi-objective example with <c>SetConstraint</c> and
    ///     <see cref="NSGAIISampler" /> must reproduce Optuna's full ledger — including the two constraint values
    ///     each trial records in its system attributes, the deep proof that constrained NSGA-II matches.
    /// </summary>
    /// <param name="seed">NSGAIISampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Constraints_BinhKorn_NsgaII_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            sampler: new NSGAIISampler(seed: seed), studyName: "faq-con",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        cs.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -15, 30);
            double y = t.SuggestFloat("y", -15, 30);
            double c0 = (x - 5.0) * (x - 5.0) + y * y - 25.0;
            double c1 = -((x - 8.0) * (x - 8.0)) - (y + 3.0) * (y + 3.0) + 7.7;
            t.SetConstraint("c0", c0);
            t.SetConstraint("c1", c1);
            double v0 = 4.0 * x * x + 4.0 * y * y;
            double v1 = (x - 5.0) * (x - 5.0) + (y - 5.0) * (y - 5.0);
            return (IReadOnlyList<double>)new[] { v0, v1 };
        }, nTrials: 32);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -15, 30)\n" +
            "    y = t.suggest_float('y', -15, 30)\n" +
            "    c0 = (x - 5.0) * (x - 5.0) + y * y - 25.0\n" +
            "    c1 = -((x - 8.0) * (x - 8.0)) - (y + 3.0) * (y + 3.0) + 7.7\n" +
            "    t.set_constraint('c0', c0)\n" +
            "    t.set_constraint('c1', c1)\n" +
            "    v0 = 4.0 * x * x + 4.0 * y * y\n" +
            "    v1 = (x - 5.0) * (x - 5.0) + (y - 5.0) * (y - 5.0)\n" +
            "    return (v0, v1)\n" +
            "study = optuna.create_study(study_name='faq-con', directions=['minimize', 'minimize'], " +
            $"sampler=optuna.samplers.NSGAIISampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=32)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== testing objectives
    // "How can I test my objective functions?"

    /// <summary>
    ///     FixedTrial answer: the FAQ's <c>x + y</c> objective, typed against <see cref="BaseTrial" /> so a
    ///     <see cref="FixedTrial" /> drives it, must return the documented values (0.0 and -5.0) and match Optuna's
    ///     <c>FixedTrial</c> replay exactly.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Testing_FixedTrial_ReplaysDocumentedValues_Parity()
    {
        PythonSession.EnsureOrInconclusive();

        PyExec(
            "from optuna.trial import FixedTrial\n" +
            "def _obj(trial):\n" +
            "    x = trial.suggest_float('x', -1.0, 1.0)\n" +
            "    y = trial.suggest_int('y', -5, 5)\n" +
            "    return x + y\n" +
            "r1 = float(_obj(FixedTrial({'x': 1.0, 'y': -1})))\n" +
            "r2 = float(_obj(FixedTrial({'x': -1.0, 'y': -4})))\n");
        double py1 = PyFloat("r1");
        double py2 = PyFloat("r2");

        // Typed against BaseTrial, exactly as the doc prescribes, so a FixedTrial can drive it.
        static double Objective(BaseTrial trial)
        {
            double x = trial.SuggestFloat("x", -1.0, 1.0);
            long y = trial.SuggestInt("y", -5, 5);
            return x + y;
        }

        double cs1 = Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = 1.0, ["y"] = -1L }));
        double cs2 = Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = -1.0, ["y"] = -4L }));

        cs1.Should().Be(0.0, "the doc states this returns 0.0");
        cs2.Should().Be(-5.0, "the doc states this returns -5.0");
        cs1.Should().Be(py1);
        cs2.Should().Be(py2);
    }

    // ================================================================== ignore duplicated samples
    // "How can I ignore duplicated samples?"

    /// <summary>Structural equality of two parameter dictionaries by name and value (the doc's <c>ParamsEqual</c>).</summary>
    /// <param name="a">First parameter map.</param>
    /// <param name="b">Second parameter map.</param>
    /// <returns>True when both maps hold the same keys mapped to equal values.</returns>
    private static bool ParamsEqual(IReadOnlyDictionary<string, object?> a, IReadOnlyDictionary<string, object?> b)
        => a.Count == b.Count && a.All(kv => b.TryGetValue(kv.Key, out var v) && Equals(kv.Value, v));

    /// <summary>
    ///     Dedup answer: an objective that reuses an earlier completed trial's value when the sampled
    ///     <c>(x, y)</c> repeat must reproduce Optuna's ledger bit-for-bit — the repeated trials carry the reused
    ///     value, and over an 11×11 grid in 100 trials duplicates certainly occur.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void IgnoreDuplicatedSamples_ReusesValue_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-dedup");
        cs.Optimize(t =>
        {
            long x = t.SuggestInt("x", -5, 5);
            long y = t.SuggestInt("y", -5, 5);
            var considered = t.Study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
            foreach (var prior in Enumerable.Reverse(considered))
                if (ParamsEqual(t.Params, prior.Params))
                    return prior.Value!.Value;   // reuse the value for the duplicated params
            return x * x + y * y;                // 2D sphere for a fresh point
        }, nTrials: 100);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            "from optuna.trial import TrialState\n" +
            "def _obj(trial):\n" +
            "    x = trial.suggest_int('x', -5, 5)\n" +
            "    y = trial.suggest_int('y', -5, 5)\n" +
            "    considered = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))\n" +
            "    for t in reversed(considered):\n" +
            "        if trial.params == t.params:\n" +
            "            return t.value\n" +
            "    return x * x + y * y\n" +
            $"study = optuna.create_study(study_name='faq-dedup', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=100)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== permutation as a parameter
    // "How can I deal with permutation as a parameter?"

    /// <summary>Euclidean city coordinates for the FAQ's Lehmer-code TSP example (6 cities).</summary>
    private static readonly double[][] TspCities =
    {
        new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 },
        new[] { 1.0, 1.0 }, new[] { 2.0, 2.0 }, new[] { -1.0, -1.0 },
    };

    /// <summary>Decodes a Lehmer code into a permutation of <c>0..n-1</c> (the doc's <c>Decode</c>).</summary>
    /// <param name="lehmerCode">Per-index counts, each within its shrinking remaining range.</param>
    /// <returns>The permutation the code encodes.</returns>
    private static int[] TspDecode(int[] lehmerCode)
    {
        int n = TspCities.Length;
        var remaining = Enumerable.Range(0, n).ToList();
        var output = new int[lehmerCode.Length];
        for (int i = 0; i < lehmerCode.Length; i++)
        {
            int value = remaining[lehmerCode[i]];
            output[i] = value;
            remaining.Remove(value);
        }
        return output;
    }

    /// <summary>
    ///     Permutation answer: re-parametrizing a permutation as an independent integer search space via a Lehmer
    ///     code must reproduce Optuna's ledger bit-for-bit — including the closed-tour length, proving the
    ///     <c>Math.Sqrt</c> distance and the wrap-around index match Python's spelling.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Permutation_LehmerCodeTsp_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        int n = TspCities.Length;
        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-perm");
        cs.Optimize(t =>
        {
            var lehmerCode = new int[n];
            for (int i = 0; i < n; i++)
                lehmerCode[i] = (int)t.SuggestInt($"x{i}", 0, n - i - 1);
            int[] permutation = TspDecode(lehmerCode);

            double total = 0.0;
            for (int i = 0; i < n; i++)
            {
                double[] a = TspCities[permutation[i]];
                double[] b = TspCities[permutation[(i - 1 + n) % n]];   // np.roll(perm, 1) equivalent
                total += Math.Sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
            }
            return total;
        }, nTrials: 20);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            "import math\n" +
            "city = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]]\n" +
            "n = len(city)\n" +
            "def _decode(code):\n" +
            "    remaining = list(range(n))\n" +
            "    out = []\n" +
            "    for k in code:\n" +
            "        v = remaining[k]\n" +
            "        out.append(v)\n" +
            "        remaining.remove(v)\n" +
            "    return out\n" +
            "def _obj(t):\n" +
            "    code = [t.suggest_int(f'x{i}', 0, n - i - 1) for i in range(n)]\n" +
            "    perm = _decode(code)\n" +
            "    total = 0.0\n" +
            "    for i in range(n):\n" +
            "        a = city[perm[i]]\n" +
            "        b = city[perm[(i - 1 + n) % n]]\n" +
            "        total += math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))\n" +
            "    return total\n" +
            $"study = optuna.create_study(study_name='faq-perm', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=20)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== MySQL case sensitivity
    // "How can I resolve case sensitivity issues with MySQL?"

    /// <summary>
    ///     Case-sensitivity answer: OptunaSharp (like Optuna) treats parameter names case-sensitively, so an
    ///     objective suggesting both <c>a</c> and <c>A</c> registers two distinct parameters — reproduced exactly
    ///     against Optuna in memory (the collation caveat is a MySQL property, not an OptunaSharp one).
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void CaseSensitivity_DistinctParams_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new RandomSampler(seed), studyName: "faq-case");
        cs.Optimize(t =>
        {
            long a = t.SuggestInt("a", 0, 10);
            long aUpper = t.SuggestInt("A", 0, 10);
            return a + aUpper;
        }, nTrials: 10);
        var csState = StudySnapshot.Capture(cs);
        csState.Trials[0].Params.Keys.Should().Contain(new[] { "a", "A" }, "both case variants are distinct params");

        PyExec(
            "def _obj(t):\n" +
            "    a = t.suggest_int('a', 0, 10)\n" +
            "    A = t.suggest_int('A', 0, 10)\n" +
            "    return a + A\n" +
            $"study = optuna.create_study(study_name='faq-case', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_obj, n_trials=10)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================== save and resume (SQLite RDB)
    // "Can I use OptunaSharp without remote RDB servers?" / "How can I save and resume studies?"

    /// <summary>A unique temp SQLite path, forward-slashed for the <c>sqlite:///</c> URL on both runtimes.</summary>
    /// <returns>An absolute <c>.db</c> path that does not yet exist.</returns>
    private static string TempDbPath()
        => Path.Combine(Path.GetTempPath(), $"oss-faq-{Guid.NewGuid():N}.db").Replace('\\', '/');

    /// <summary>Best-effort removal of a SQLite file and any WAL/SHM siblings it left behind.</summary>
    /// <param name="path">The <c>.db</c> path.</param>
    private static void DeleteDb(string path)
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
                // A connection may briefly hold the file; leaving a temp file behind is harmless.
            }
        }
    }

    /// <summary>
    ///     Save/resume answer: creating a study on a SQLite <see cref="RDBStorage" />, closing it, then re-opening
    ///     by name with <c>loadIfExists</c> and optimizing more must reproduce Optuna's equivalent two-phase run
    ///     bit-for-bit. Each phase uses a fresh (re-seeded) sampler on both sides — exactly the doc's "storage does
    ///     not persist sampler state" note — and the 20-trial resumed study matches.
    /// </summary>
    /// <param name="seed">Seed for phase 1; phase 2 uses <paramref name="seed" /> + 1 on both sides.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SaveResume_SqliteRdb_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        string csPath = TempDbPath();
        string pyPath = TempDbPath();
        try
        {
            // Phase 1: create + optimize, then dispose the storage so the file is closed and re-openable.
            using (var s1 = new RDBStorage($"sqlite:///{csPath}"))
            {
                var st1 = Optuna.CreateStudy(
                    storage: s1, studyName: "faq-resume", sampler: new RandomSampler(seed));
                st1.Optimize(Quadratic, nTrials: 10);
            }

            // Phase 2: re-open by name with loadIfExists + a fresh sampler, optimize 10 more, capture.
            StudySnapshot csState;
            using (var s2 = new RDBStorage($"sqlite:///{csPath}"))
            {
                var st2 = Optuna.CreateStudy(
                    storage: s2, studyName: "faq-resume", sampler: new RandomSampler(seed + 1), loadIfExists: true);
                st2.Optimize(Quadratic, nTrials: 10);
                csState = StudySnapshot.Capture(st2);
            }
            csState.Trials.Count.Should().Be(20, "10 initial + 10 resumed trials");

            PyExec(
                PyQuadratic +
                $"study = optuna.create_study(study_name='faq-resume', storage='sqlite:///{pyPath}', " +
                $"sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
                "study.optimize(_q, n_trials=10)\n" +
                $"study = optuna.create_study(study_name='faq-resume', storage='sqlite:///{pyPath}', " +
                $"sampler=optuna.samplers.RandomSampler(seed={seed + 1}), load_if_exists=True)\n" +
                "study.optimize(_q, n_trials=10)\n");

            StudySnapshot.AssertParity(PyStudyState("study"), csState);
        }
        finally
        {
            DeleteDb(csPath);
            DeleteDb(pyPath);
        }
    }
}

using System.Collections.Generic;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna deep parity for the ablation-study recipe (<c>recipes/ablation-study.md</c>), which
///     drives an exhaustive sweep with <see cref="BruteForceSampler" />. Under a shared seed the sampler
///     enumerates the (conditional) search space in the same order on both runtimes, so the entire trial
///     ledger — every combination, its params, and its recorded value — must match live Optuna
///     bit-for-bit. Objective values are computed from numeric parameters only (never a formatted config
///     string, whose float formatting differs between C# and Python) so the recorded values stay bit-exact.
/// </summary>
[TestClass]
public class RecipeBruteForceParityTests : OptunaInteropTestBase
{
    // ---------------------------------------------------------------- basic ablation

    /// <summary>Flat three-factor grid: optimizer × stepped learning rate × schedule.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>A deterministic score over the grid point.</returns>
    private static double AblationBasic(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "Adam", "SGD", "RMSprop" });
        double lr = t.SuggestFloat("lr", 0.001, 0.01, step: 0.001);
        string lrSchedule = t.SuggestCategorical("lr_schedule", new[] { "constant", "cosine" });

        var mock = new Dictionary<string, double> { ["Adam"] = 0.9, ["SGD"] = 0.85, ["RMSprop"] = 0.88 };
        return mock[optimizer] + lr * 10 + (lrSchedule == "cosine" ? 0.01 : 0.0);
    }

    private const string PyAblationBasic =
        "def _obj(t):\n" +
        "    optimizer = t.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])\n" +
        "    lr = t.suggest_float('lr', 0.001, 0.01, step=0.001)\n" +
        "    lr_schedule = t.suggest_categorical('lr_schedule', ['constant', 'cosine'])\n" +
        "    mock = {'Adam': 0.9, 'SGD': 0.85, 'RMSprop': 0.88}\n" +
        "    return mock[optimizer] + lr * 10 + (0.01 if lr_schedule == 'cosine' else 0.0)\n";

    /// <summary>
    ///     Basic ablation: a 3×10×2 = 60-point grid enumerated exhaustively by BruteForceSampler must
    ///     reproduce Optuna's full trial ledger (order, params, values) exactly.
    /// </summary>
    /// <param name="seed">BruteForceSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Ablation_BruteForceBasic_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new BruteForceSampler(seed), studyName: "ab");
        cs.Optimize(AblationBasic);   // no nTrials: BruteForce stops when the grid is exhausted
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='ab', direction='maximize', sampler=optuna.samplers.BruteForceSampler(seed={seed}))\n" +
            PyAblationBasic +
            "study.optimize(_obj)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ---------------------------------------------------------------- conditional ablation

    /// <summary>Conditional grid: Adam exposes beta1/beta2, SGD exposes momentum/nesterov; lr is shared.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>A deterministic score built from numeric params (branch-dependent).</returns>
    private static double AblationConditional(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "Adam", "SGD" });
        double beta1 = 0, beta2 = 0, momentum = 0;
        bool nesterov = false;
        if (optimizer == "Adam")
        {
            beta1 = t.SuggestCategorical("beta1", new[] { 0.9, 0.95 });
            beta2 = t.SuggestCategorical("beta2", new[] { 0.999, 0.9999 });
        }
        else
        {
            momentum = t.SuggestCategorical("momentum", new[] { 0.0, 0.9, 0.99 });
            nesterov = t.SuggestCategorical("nesterov", new[] { true, false });
        }
        double lr = t.SuggestFloat("lr", 0.001, 0.01, step: 0.001);
        return optimizer == "Adam"
            ? beta1 * 0.1 + beta2 * 0.01 + lr
            : momentum * 0.1 + (nesterov ? 0.02 : 0.0) + lr;
    }

    private const string PyAblationConditional =
        "def _obj(t):\n" +
        "    optimizer = t.suggest_categorical('optimizer', ['Adam', 'SGD'])\n" +
        "    if optimizer == 'Adam':\n" +
        "        beta1 = t.suggest_categorical('beta1', [0.9, 0.95])\n" +
        "        beta2 = t.suggest_categorical('beta2', [0.999, 0.9999])\n" +
        "    else:\n" +
        "        momentum = t.suggest_categorical('momentum', [0.0, 0.9, 0.99])\n" +
        "        nesterov = t.suggest_categorical('nesterov', [True, False])\n" +
        "    lr = t.suggest_float('lr', 0.001, 0.01, step=0.001)\n" +
        "    if optimizer == 'Adam':\n" +
        "        return beta1 * 0.1 + beta2 * 0.01 + lr\n" +
        "    return momentum * 0.1 + (0.02 if nesterov else 0.0) + lr\n";

    /// <summary>
    ///     Conditional ablation: (4 Adam + 6 SGD) branches × 10 learning rates = 100 trials. BruteForce's
    ///     define-by-run enumeration of the hierarchical space must match Optuna's exactly, trial-for-trial.
    /// </summary>
    /// <param name="seed">BruteForceSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Ablation_BruteForceConditional_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new BruteForceSampler(seed), studyName: "abc");
        cs.Optimize(AblationConditional);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='abc', direction='maximize', sampler=optuna.samplers.BruteForceSampler(seed={seed}))\n" +
            PyAblationConditional +
            "study.optimize(_obj)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }
}

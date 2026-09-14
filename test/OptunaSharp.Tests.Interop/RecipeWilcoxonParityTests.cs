using OptunaSharp.Collections;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna deep parity for the Wilcoxon-pruner recipe (<c>recipes/wilcoxon-pruner.md</c>). The
///     objective averages many per-instance evaluations and reports each one, so
///     <see cref="WilcoxonPruner" /> can early-stop trials that are statistically worse than the best. Under
///     a shared seed, OptunaSharp's pruner must make the same signed-rank prune decisions at the same steps
///     as real Optuna — asserted through the full trial ledger (a diverging decision would change how many
///     intermediate values a trial records).
///
///     <para>Per-instance costs are <c>BaseCost[i] · (1 + a)</c> so trial-to-trial differences share one
///     sign (a clean, decisive signed-rank test, away from the p-threshold knife-edge), and the reported
///     mean uses an explicit left-to-right sum — NOT <c>.Average()</c> / Python <c>sum()</c>, whose 3.12
///     compensated summation would drift ~1 ULP — keeping every recorded value bit-exact.</para>
/// </summary>
[TestClass]
public class RecipeWilcoxonParityTests : OptunaInteropTestBase
{
    private const int NInstances = 20;

    /// <summary>
    ///     Averages 20 per-instance costs, reporting each; on a prune signal it returns the running mean
    ///     (the recipe's workaround that feeds the sampler an estimate instead of raising).
    /// </summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The mean per-instance cost (full or running-on-prune).</returns>
    private static double WilcoxonObjective(Trial t)
    {
        double a = t.SuggestFloat("a", 0.0, 2.0);
        double sum = 0;
        int n = 0;
        for (int i = 0; i < NInstances; i++)
        {
            double cost = (1.0 + i * 0.1) * (1.0 + a);   // BaseCost[i] * (1 + a); bit-exact
            sum += cost;
            n++;
            t.Report(cost, i);
            if (t.ShouldPrune())
                return sum / n;   // running mean via explicit sum (no compensated summation)
        }
        return sum / n;
    }

    private const string PyWilcoxonObjective =
        "def _obj(t):\n" +
        "    a = t.suggest_float('a', 0.0, 2.0)\n" +
        "    s = 0.0\n" +
        "    n = 0\n" +
        "    for i in range(20):\n" +
        "        cost = (1.0 + i * 0.1) * (1.0 + a)\n" +
        "        s += cost\n" +
        "        n += 1\n" +
        "        t.report(cost, i)\n" +
        "        if t.should_prune():\n" +
        "            return s / n\n" +
        "    return s / n\n";

    /// <summary>
    ///     Wilcoxon-pruner recipe: a TPE-free (RandomSampler-seeded) run with an enqueued baseline must
    ///     prune exactly the same instance-averaged trials as live Optuna's <c>WilcoxonPruner(p=0.1)</c>,
    ///     with an identical trial ledger (states, per-instance intermediate values, and values).
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Wilcoxon_InstanceAveraged_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Minimize, sampler: new RandomSampler(seed),
            pruner: new WilcoxonPruner(pThreshold: 0.1), studyName: "wx");
        cs.EnqueueTrial(new OrderedDict<string, object?> { ["a"] = 1.0 });   // baseline first
        cs.Optimize(WilcoxonObjective, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(direction='minimize', study_name='wx', sampler=optuna.samplers.RandomSampler(seed={seed}), pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1))\n" +
            "study.enqueue_trial({'a': 1.0})\n" +
            PyWilcoxonObjective +
            "study.optimize(_obj, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }
}

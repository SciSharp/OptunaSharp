using System;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna deep parity for the two "roll your own" recipes: a user-defined
///     <see cref="BasePruner" /> (<c>recipes/user-defined-pruner.md</c>) and a user-defined
///     <see cref="BaseSampler" /> (<c>recipes/user-defined-sampler.md</c>). Each is implemented
///     identically in C# and in Python (subclassing the same Optuna base classes), driven by a shared
///     seed and a deterministic objective, and the full trial ledger is asserted bit-for-bit against real
///     Optuna — proving the extension points behave identically down to the RNG.
/// </summary>
[TestClass]
public class RecipeUserDefinedParityTests : OptunaInteropTestBase
{
    // ================================================================ user-defined-pruner.md

    /// <summary>The recipe's LastPlacePruner: prunes a trial in last place among peers at the same step.</summary>
    private sealed class LastPlacePruner : BasePruner
    {
        private readonly int _warmupSteps;
        private readonly int _warmupTrials;

        /// <param name="warmupSteps">Minimum step before pruning is considered.</param>
        /// <param name="warmupTrials">Minimum peer count before pruning is considered.</param>
        public LastPlacePruner(int warmupSteps, int warmupTrials)
        {
            _warmupSteps = warmupSteps;
            _warmupTrials = warmupTrials;
        }

        /// <inheritdoc />
        public override bool Prune(Study study, FrozenTrial trial)
        {
            long? step = trial.LastStep;
            if (step is null)
                return false;

            double thisScore = trial.IntermediateValues[step.Value];
            var completed = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
            var otherScores = completed
                .Where(t => t.IntermediateValues.ContainsKey(step.Value))
                .Select(t => t.IntermediateValues[step.Value])
                .OrderBy(s => s)
                .ToList();

            if (step.Value >= _warmupSteps && otherScores.Count > _warmupTrials)
                if (thisScore < otherScores[0])
                    return true;
            return false;
        }
    }

    /// <summary>Deterministic five-step objective whose trajectory depends on the sampled <c>alpha</c>.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The final (or last-reported) score; larger is better.</returns>
    private static double LastPlaceObjective(Trial t)
    {
        double alpha = t.SuggestFloat("alpha", 1e-5, 1e-3, log: true);
        double score = 0;
        for (int step = 0; step < 5; step++)
        {
            score = step * (0.1 - alpha * 1000.0);   // bit-exact; some alphas trend down (last place)
            t.Report(score, step);
            if (t.ShouldPrune())
                throw new TrialPruned();
        }
        return score;
    }

    private const string PyLastPlace =
        "class LastPlacePruner(optuna.pruners.BasePruner):\n" +
        "    def __init__(self, warmup_steps, warmup_trials):\n" +
        "        self._warmup_steps = warmup_steps\n" +
        "        self._warmup_trials = warmup_trials\n" +
        "    def prune(self, study, trial):\n" +
        "        step = trial.last_step\n" +
        "        if step is None:\n" +
        "            return False\n" +
        "        this_score = trial.intermediate_values[step]\n" +
        "        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))\n" +
        "        other = sorted(t.intermediate_values[step] for t in completed if step in t.intermediate_values)\n" +
        "        if step >= self._warmup_steps and len(other) > self._warmup_trials:\n" +
        "            if this_score < other[0]:\n" +
        "                return True\n" +
        "        return False\n" +
        "def _obj(t):\n" +
        "    alpha = t.suggest_float('alpha', 1e-5, 1e-3, log=True)\n" +
        "    score = 0.0\n" +
        "    for step in range(5):\n" +
        "        score = step * (0.1 - alpha * 1000.0)\n" +
        "        t.report(score, step)\n" +
        "        if t.should_prune():\n" +
        "            raise optuna.TrialPruned()\n" +
        "    return score\n";

    /// <summary>
    ///     User-defined pruner: the LastPlacePruner (identical logic in C# and Python) must prune exactly
    ///     the same trials at the same steps as real Optuna, and the whole study ledger must agree.
    /// </summary>
    /// <param name="seed">RandomSampler seed shared by both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void UserDefinedPruner_LastPlace_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomSampler(seed),
            pruner: new LastPlacePruner(warmupSteps: 1, warmupTrials: 5), studyName: "lp");
        cs.Optimize(LastPlaceObjective, nTrials: 30);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            PyLastPlace +
            $"study = optuna.create_study(direction='maximize', study_name='lp', sampler=optuna.samplers.RandomSampler(seed={seed}), pruner=LastPlacePruner(warmup_steps=1, warmup_trials=5))\n" +
            "study.optimize(_obj, n_trials=30)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }

    // ================================================================ user-defined-sampler.md

    /// <summary>
    ///     The recipe's Simulated-Annealing sampler. For a deterministic parity test both the SA rng and the
    ///     independent-sampling rng are explicitly seeded (the recipe leaves them default/unseeded); this is
    ///     the only change from the doc snippet, and it is applied identically on the Python side.
    /// </summary>
    private sealed class SimulatedAnnealingSampler : BaseSampler
    {
        private readonly LazyRandomState _rng;
        private readonly RandomSampler _independent;
        private double _temperature;
        private FrozenTrial? _currentTrial;

        /// <param name="seed">Seed for both the SA neighborhood draws and the independent fallback sampler.</param>
        /// <param name="temperature">Initial annealing temperature.</param>
        public SimulatedAnnealingSampler(int seed, double temperature = 100)
        {
            _rng = new LazyRandomState(seed);
            _independent = new RandomSampler(seed);
            _temperature = temperature;
        }

        /// <inheritdoc />
        public override OrderedDict<string, object?> SampleRelative(
            Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        {
            if (searchSpace.Count == 0)
                return new OrderedDict<string, object?>();

            var trials = study.GetTrials(deepcopy: false);
            FrozenTrial prevTrial = trials[^2];
            double probability = 1.0;
            if (_currentTrial is not null && prevTrial.Value > _currentTrial.Value)
                probability = Math.Exp((_currentTrial.Value!.Value - prevTrial.Value!.Value) / _temperature);
            _temperature *= 0.9;

            if ((double)_rng.Rng.uniform(0.0, 1.0) < probability)
                _currentTrial = prevTrial;

            var sampled = new OrderedDict<string, object?>();
            foreach (var (name, dist) in searchSpace)
            {
                var fd = (FloatDistribution)dist;
                double current = (double)_currentTrial!.Params[name]!;
                double width = (fd.High - fd.Low) * 0.1;
                double low = Math.Max(current - width, fd.Low);
                double high = Math.Min(current + width, fd.High);
                sampled[name] = (double)_rng.Rng.uniform(low, high);
            }
            return sampled;
        }

        /// <inheritdoc />
        public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
            => SearchSpaceFunctions.IntersectionSearchSpaceOf(study.GetTrials(deepcopy: false));

        /// <inheritdoc />
        public override object? SampleIndependent(
            Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
            => _independent.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    private const string PySimulatedAnnealing =
        "class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):\n" +
        "    def __init__(self, seed, temperature=100):\n" +
        "        self._rng = np.random.RandomState(seed)\n" +
        "        self._independent = optuna.samplers.RandomSampler(seed)\n" +
        "        self._temperature = temperature\n" +
        "        self._current_trial = None\n" +
        "    def sample_relative(self, study, trial, search_space):\n" +
        "        if search_space == {}:\n" +
        "            return {}\n" +
        "        prev_trial = study.get_trials(deepcopy=False)[-2]\n" +
        "        if self._current_trial is None or prev_trial.value <= self._current_trial.value:\n" +
        "            probability = 1.0\n" +
        "        else:\n" +
        "            probability = np.exp((self._current_trial.value - prev_trial.value) / self._temperature)\n" +
        "        self._temperature *= 0.9\n" +
        "        if self._rng.uniform(0, 1) < probability:\n" +
        "            self._current_trial = prev_trial\n" +
        "        params = {}\n" +
        "        for param_name, param_distribution in search_space.items():\n" +
        "            current_value = self._current_trial.params[param_name]\n" +
        "            width = (param_distribution.high - param_distribution.low) * 0.1\n" +
        "            neighbor_low = max(current_value - width, param_distribution.low)\n" +
        "            neighbor_high = min(current_value + width, param_distribution.high)\n" +
        "            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)\n" +
        "        return params\n" +
        "    def infer_relative_search_space(self, study, trial):\n" +
        "        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))\n" +
        "    def sample_independent(self, study, trial, param_name, param_distribution):\n" +
        "        return self._independent.sample_independent(study, trial, param_name, param_distribution)\n";

    /// <summary>Deterministic two-variable objective (<c>x² + y</c>), the recipe's example.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The objective value.</returns>
    private static double SaObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        double y = t.SuggestFloat("y", -5, 5);
        return x * x + y;
    }

    private const string PySaObjective =
        "def _obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    y = t.suggest_float('y', -5, 5)\n" +
        "    return x * x + y\n";

    /// <summary>
    ///     User-defined sampler: the Simulated-Annealing sampler (identical in C# and Python, both seeded)
    ///     must reproduce Optuna's run exactly — this validates the BaseSampler relative/independent split,
    ///     the intersection search space, and that <c>LazyRandomState</c>'s draws are bit-identical to
    ///     NumPy's <c>RandomState</c> through a real user sampler.
    /// </summary>
    /// <param name="seed">Seed shared by the SA rng and independent sampler on both runtimes.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1)]
    [DataRow(7)]
    public void UserDefinedSampler_SimulatedAnnealing_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(sampler: new SimulatedAnnealingSampler(seed), studyName: "sa");
        cs.Optimize(SaObjective, nTrials: 40);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            "import numpy as np\n" +
            PySimulatedAnnealing +
            PySaObjective +
            $"study = optuna.create_study(study_name='sa', sampler=SimulatedAnnealingSampler({seed}))\n" +
            "study.optimize(_obj, n_trials=40)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState);
    }
}

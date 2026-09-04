using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-for-byte parity of the ported <c>ParamRepeatPruner</c> (community gist
///     aisu-wata0/24045ab5e8ad007b4a09f708adfe359f, 6★) against the original Python class. Both run the
///     same seeded study; the pruner prunes any trial whose params repeat an earlier COMPLETE trial's,
///     so under a bit-exact param trajectory the set of PRUNED vs COMPLETE trials must match exactly.
///
///     <para>Uses <see cref="RandomSampler" /> (not the gist's TPE): the param trajectory is then
///     value-independent and bit-identical to Python, so the repeats — and the pruning decisions they
///     drive — are byte-comparable. TPE over an integer parameter with repeats is deliberately excluded
///     (NumPy x86-simd-sort tie order; see the testing skill). The Python side embeds the gist's class
///     reduced to its default path (repeats_max=0, compare against COMPLETE).</para>
/// </summary>
[TestClass]
public class ParamRepeatPrunerParityTests : OptunaInteropTestBase
{
    // The gist's ParamRepeatPruner, default path (repeats_max=0, should_compare_states=[COMPLETE]).
    private const string PyPruner =
        "from optuna.trial import TrialState\n" +
        "class ParamRepeatPruner:\n" +
        "    def __init__(self, study, repeats_max=0, should_compare_states=(TrialState.COMPLETE,)):\n" +
        "        self.study = study\n" +
        "        self.repeats_max = repeats_max\n" +
        "        self.should_compare_states = should_compare_states\n" +
        "        self.repeats = {}\n" +
        "    def check_params(self):\n" +
        "        trials = self.study.trials\n" +
        "        current = trials[-1]\n" +
        "        repeated_number = -1\n" +
        "        for past in trials[:-1]:\n" +
        "            if past.state not in self.should_compare_states:\n" +
        "                continue\n" +
        "            if current.params == past.params:\n" +
        "                repeated_number = past.number\n" +
        "                break\n" +
        "        if repeated_number > -1:\n" +
        "            self.repeats.setdefault(repeated_number, []).append(current.number)\n" +
        "            if len(self.repeats[repeated_number]) > self.repeats_max:\n" +
        "                raise optuna.exceptions.TrialPruned()\n" +
        "        return repeated_number\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void RepeatedParamsPruned_ByteForByte(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new RandomSampler(seed),
            direction: StudyDirection.Minimize, studyName: "param-repeat");
        var pruner = new PortedParamRepeatPruner(study);
        study.Optimize(t =>
        {
            long p = t.SuggestInt("dummy_param-0", 1, 20);
            pruner.CheckParams(t);
            return p;
        }, nTrials: 40);
        var csState = StudySnapshot.Capture(study);

        PyExec(PyPruner +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}), " +
            "direction='minimize', study_name='param-repeat')\n" +
            "pruner = ParamRepeatPruner(study)\n" +
            "def _obj(t):\n" +
            "    p = t.suggest_int('dummy_param-0', 1, 20)\n" +
            "    pruner.check_params()\n" +
            "    return p\n" +
            "study.optimize(_obj, n_trials=40)\n");
        var pyState = PyStudyState("study");

        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
        csState.Trials.Count(t => t.State == (long)TrialState.Pruned).Should().BeGreaterThan(0,
            "int(1,20) over 40 trials must produce repeats that get pruned — else the test proves nothing");
    }

    // A standalone copy of the port in examples/ParamRepeatPruner.cs (a userland pattern, not Core).
    private sealed class PortedParamRepeatPruner
    {
        private readonly Study _study;
        private readonly int _repeatsMax;
        private readonly HashSet<TrialState> _compareStates;
        private readonly Dictionary<int, List<int>> _repeats = new();

        public PortedParamRepeatPruner(Study study, int repeatsMax = 0)
        {
            _study = study;
            _repeatsMax = repeatsMax;
            _compareStates = new HashSet<TrialState> { TrialState.Complete };
        }

        public int CheckParams(Trial current)
        {
            int repeated = -1;
            foreach (var past in _study.GetTrials(deepcopy: false))
            {
                if (past.Number == current.Number || !_compareStates.Contains(past.State))
                    continue;
                if (ParamsEqual(current.Params, past.Params))
                {
                    repeated = past.Number;
                    break;
                }
            }
            if (repeated > -1)
            {
                if (!_repeats.TryGetValue(repeated, out var seen))
                    _repeats[repeated] = seen = new List<int>();
                seen.Add(current.Number);
                if (seen.Count > _repeatsMax)
                    throw new TrialPruned();
            }
            return repeated;
        }

        private static bool ParamsEqual(IReadOnlyDictionary<string, object?> a, IReadOnlyDictionary<string, object?> b)
        {
            if (a.Count != b.Count)
                return false;
            foreach (var (key, av) in a)
                if (!b.TryGetValue(key, out var bv) || !Equals(av, bv))
                    return false;
            return true;
        }
    }
}

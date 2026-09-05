using AwesomeAssertions;
using OptunaSharp.Collections;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna parity for <see cref="FixedTrial" /> (plan 4.9.3) — the study-free
///     <c>BaseTrial</c> that replays a fixed <c>params</c> dict. Running the same objective over
///     the same fixed values in real Optuna and OptunaSharp must return a bit-identical objective
///     value (the objectives use <c>x * x</c>, not <c>x ** 2</c>, so the arithmetic stays
///     bit-identical across the boundary), and an out-of-range fixed value must warn yet still
///     return verbatim in both.
/// </summary>
[TestClass]
public class FixedTrialParityTests : OptunaInteropTestBase
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FixedTrial_FloatCategoricalInt_MatchesOptuna()
    {
        PyExec(
            "from optuna.trial import FixedTrial\n" +
            "def _obj(trial):\n" +
            "    x = trial.suggest_float('x', -100, 100)\n" +
            "    y = trial.suggest_categorical('y', [-1, 0, 1])\n" +
            "    z = trial.suggest_int('z', 0, 10)\n" +
            "    return x * x + y + z\n" +
            "result = float(_obj(FixedTrial({'x': 1.5, 'y': 0, 'z': 3})))\n");
        double pyResult = PyFloat("result");

        var fixedTrial = new FixedTrial(new OrderedDict<string, object?> { ["x"] = 1.5, ["y"] = 0L, ["z"] = 3L });
        double Objective(BaseTrial trial)
        {
            double x = trial.SuggestFloat("x", -100, 100);
            long y = trial.SuggestCategorical("y", new[] { -1, 0, 1 });
            long z = trial.SuggestInt("z", 0, 10);
            return x * x + y + z;
        }

        double csResult = Objective(fixedTrial);

        csResult.Should().Be(pyResult);
        fixedTrial.Params.Keys.Should().Equal("x", "y", "z");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FixedTrial_OutOfRangeValue_WarnsButReturnsVerbatim_MatchesOptuna()
    {
        PyExec(
            "import warnings\n" +
            "warnings.simplefilter('ignore')\n" +
            "from optuna.trial import FixedTrial\n" +
            "def _obj2(trial):\n" +
            "    return trial.suggest_float('x', 0.0, 1.0)\n" +
            "result2 = float(_obj2(FixedTrial({'x': 5.0})))\n");
        double pyResult = PyFloat("result2");

        // 5.0 is outside [0, 1]: Optuna and OptunaSharp both warn and return the fixed value verbatim.
        var fixedTrial = new FixedTrial(new OrderedDict<string, object?> { ["x"] = 5.0 });
        double csResult = fixedTrial.SuggestFloat("x", 0.0, 1.0);

        pyResult.Should().Be(5.0);
        csResult.Should().Be(pyResult);
    }
}

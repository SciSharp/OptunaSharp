using System;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Parity scenarios distilled from <b>battle-tested, in-the-wild Optuna usage</b> — patterns
///     harvested from the most-starred public repos that <c>import optuna</c> (GitHub code search)
///     and from the Optuna maintainers' own gists — rather than from synthetic fuzzing. The intent
///     is to pin the behaviors real projects actually depend on, byte-for-byte against the vendored
///     5.0.0.dev reference.
///
///     <para><b>This suite: non-finite objective RETURN values.</b> Source: the optuna-dashboard
///     test fixture by maintainer keisuke-umezawa
///     (<c>gist d57c1a2acd4afebe3845ecec22a2b6bc</c>, "optuna sciprt") — its <c>objective_single_inf</c>
///     returns <c>+inf</c> / <c>-inf</c> / <c>x**2</c> chosen by <c>trial.number % 3</c>. The existing
///     edge-case coverage exercises <c>NaN</c>/±<c>inf</c> as <em>intermediate reports</em>
///     (<see cref="EdgeCaseParityTests" />); a non-finite value <em>returned</em> from the objective is
///     a distinct path — it is stored as the trial's value and then feeds the next suggestion, so it
///     stresses each sampler's value handling (the loss/ordering in TPE's below/above split in
///     particular).</para>
/// </summary>
[TestClass]
public class BattleTestedUsageParityTests : OptunaInteropTestBase
{
    private const double FloatTol = 1e-9;

    /// <summary>
    ///     keisuke-umezawa's fixture objective. <c>x**2</c> is written as <see cref="Math.Pow" /> (not
    ///     <c>x*x</c>) so the finite value is bit-identical to Python's <c>x ** 2</c> — see the numeric
    ///     parity notes on scalar power.
    /// </summary>
    private static double InfFixtureObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        var n = t.Number;
        if (n % 3 == 0)
            return double.PositiveInfinity;
        if (n % 3 == 1)
            return double.NegativeInfinity;
        return Math.Pow(x, 2.0);
    }

    /// <summary>The identical objective on the Python side (bound into the scope, then optimized).</summary>
    private const string PyInfFixture =
        "def _obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    if t.number % 3 == 0:\n" +
        "        return float('inf')\n" +
        "    if t.number % 3 == 1:\n" +
        "        return float('-inf')\n" +
        "    return x ** 2\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void InfNegInfReturns_RandomSampler_BitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // RandomSampler ignores trial history, so the whole study — including every stored ±inf and
        // Math.Pow(x, 2) value — must be bit-identical to Python's MT19937-seeded run.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new RandomSampler(seed), studyName: "single-inf");
        csStudy.Optimize(InfFixtureObjective, nTrials: 9);
        var csState = StudySnapshot.Capture(csStudy);

        PyExec(
            $"study = optuna.create_study(study_name='single-inf', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            PyInfFixture +
            "study.optimize(_obj, n_trials=9)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void InfNegInfReturns_Tpe_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // TPE sorts/splits completed trials by value, so the returned ±inf values participate in the
        // below/above partition. Staying in RNG lock-step (branch + state + param set exact, the
        // continuous x to the erf tolerance, non-finite values exact) proves OptunaSharp's TPE orders
        // non-finite-valued trials the same way Optuna does.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: seed), studyName: "single-inf");
        csStudy.Optimize(InfFixtureObjective, nTrials: 20);

        AssertTrajectoryParity(csStudy,
            $"study = optuna.create_study(study_name='single-inf', sampler=optuna.samplers.TPESampler(seed={seed}))\n" +
            PyInfFixture +
            "study.optimize(_obj, n_trials=20)\n");
    }

    /// <summary>
    ///     Trajectory parity for erf-routed samplers: exact branch/state/param-set and non-finite
    ///     values, continuous values to <see cref="FloatTol" />. Mirrors
    ///     <see cref="TpeSamplerParityTests" />'s comparison but compares non-finite trial values
    ///     exactly (BeApproximately is undefined at ±inf).
    /// </summary>
    private void AssertTrajectoryParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");

        csState.Trials.Count.Should().Be(pyState.Trials.Count, "trajectory length must match");
        for (int i = 0; i < pyState.Trials.Count; i++)
        {
            var e = pyState.Trials[i];
            var a = csState.Trials[i];
            a.Number.Should().Be(e.Number);
            a.State.Should().Be(e.State, $"trial {e.Number} state");
            a.Params.Keys.Should().BeEquivalentTo(e.Params.Keys, $"trial {e.Number} param set");

            foreach (var key in e.Params.Keys)
            {
                object? ev = e.Params[key];
                object? av = a.Params[key];
                if (ev is double ed && av is double ad)
                    ad.Should().BeApproximately(ed, Math.Max(FloatTol, FloatTol * Math.Abs(ed)),
                        $"trial {e.Number} param '{key}'");
                else
                    av.Should().Be(ev, $"trial {e.Number} param '{key}'");
            }

            (a.Values is null).Should().Be(e.Values is null, $"trial {e.Number} values null-ness");
            if (e.Values is not null)
            {
                a.Values!.Count.Should().Be(e.Values.Count);
                for (int k = 0; k < e.Values.Count; k++)
                {
                    double ev = e.Values[k];
                    double av = a.Values[k];
                    if (double.IsFinite(ev))
                        av.Should().BeApproximately(ev, Math.Max(FloatTol, FloatTol * Math.Abs(ev)),
                            $"trial {e.Number} values[{k}]");
                    else
                        av.Should().Be(ev, $"trial {e.Number} values[{k}] (non-finite must match exactly)");
                }
            }
        }
    }
}

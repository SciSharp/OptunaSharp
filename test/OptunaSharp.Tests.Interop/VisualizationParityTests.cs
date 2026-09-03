using System;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for Phase 3.1 part 1 — the visualization data layer builders
///     <see cref="OptimizationHistory" />, <see cref="Edf" /> and <see cref="IntermediateValues" />.
///     A seeded <see cref="RandomSampler" /> makes the trials bit-identical (MT19937) between C# and
///     Python, and every objective uses multiplication (not <c>**</c>) on both sides so the stored
///     values are bit-identical too — the scalar <c>x ** 2</c> differs from <c>x * x</c> in ~0.05% of
///     cases. The builders are pure numpy over those trials, so the derived info must match
///     <b>bit-for-bit</b> (no erf, no LAPACK anywhere on these paths).
/// </summary>
[TestClass]
public class VisualizationParityTests : OptunaInteropTestBase
{
    // ------------------------------------------------------------------ optimization history

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 50)]
    [DataRow(2026, 40)]
    public void OptimizationHistory_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        var csList = OptimizationHistory.GetInfoList(csStudy);

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n" +
            "from optuna.visualization._optimization_history import _get_optimization_history_info_list\n" +
            "il = _get_optimization_history_info_list(study, None, 'Objective Value', False)\n" +
            "info = il[0]\n");

        PyLong("len(il)").Should().Be(csList.Count);
        AssertOptimizationHistoryInfo(csList[0], "info");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 40)]
    [DataRow(13, 60)]
    public void OptimizationHistory_WithConstraints_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            t.SetConstraint("c", x);  // infeasible when x > 0.
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        var csList = OptimizationHistory.GetInfoList(csStudy);

        PyExec(
            "def _oss_obj_c(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    t.set_constraint('c', x)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_c, n_trials={nTrials})\n" +
            "from optuna.visualization._optimization_history import _get_optimization_history_info_list\n" +
            "il = _get_optimization_history_info_list(study, None, 'Objective Value', False)\n" +
            "info = il[0]\n");

        PyLong("len(il)").Should().Be(csList.Count);
        AssertOptimizationHistoryInfo(csList[0], "info");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1, 25)]
    [DataRow(99, 35)]
    public void OptimizationHistory_ErrorBar_MultiStudy_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudies = new[]
        {
            BuildQuadraticStudy("s0", seed, nTrials),
            BuildQuadraticStudy("s1", seed + 100, nTrials),
        };
        var csList = OptimizationHistory.GetInfoList(csStudies, errorBar: true);

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            "studies = []\n" +
            "for nm, sd in [('s0', " + seed + "), ('s1', " + (seed + 100) + ")]:\n" +
            "    st = optuna.create_study(study_name=nm, sampler=optuna.samplers.RandomSampler(seed=sd))\n" +
            $"    st.optimize(_oss_obj, n_trials={nTrials})\n" +
            "    studies.append(st)\n" +
            "from optuna.visualization._optimization_history import _get_optimization_history_info_list\n" +
            "il = _get_optimization_history_info_list(studies, None, 'Objective Value', True)\n" +
            "info = il[0]\n");

        PyLong("len(il)").Should().Be(csList.Count);
        var csInfo = csList[0];

        // Error-bar aggregate: trial_numbers, means, and stds (np.mean/np.std over the per-number groups).
        PyLong("len(info.trial_numbers)").Should().Be(csInfo.TrialNumbers.Count);
        for (int i = 0; i < csInfo.TrialNumbers.Count; i++)
            PyLong($"info.trial_numbers[{i}]").Should().Be(csInfo.TrialNumbers[i]);

        AssertValuesInfo(csInfo.ValuesInfo, "info.values_info", hasStds: true);
        AssertValuesInfo(csInfo.BestValuesInfo!, "info.best_values_info", hasStds: true);
    }

    // ------------------------------------------------------------------ EDF

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 45)]
    public void Edf_SingleStudy_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = BuildQuadraticStudy("edf", seed, nTrials);
        var csInfo = Edf.GetInfo(csStudy);

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(study_name='edf', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n" +
            "from optuna.visualization._edf import _get_edf_info\n" +
            "info = _get_edf_info(study)\n");

        AssertEdfInfo(csInfo, "info");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 25)]
    [DataRow(2027, 40)]
    public void Edf_MultiStudy_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudies = new[]
        {
            BuildQuadraticStudy("e0", seed, nTrials),
            BuildQuadraticStudy("e1", seed + 50, nTrials + 5),
        };
        var csInfo = Edf.GetInfo(csStudies);

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            "studies = []\n" +
            "for nm, sd, nt in [('e0', " + seed + ", " + nTrials + "), ('e1', " + (seed + 50) + ", " + (nTrials + 5) + ")]:\n" +
            "    st = optuna.create_study(study_name=nm, sampler=optuna.samplers.RandomSampler(seed=sd))\n" +
            "    st.optimize(_oss_obj, n_trials=nt)\n" +
            "    studies.append(st)\n" +
            "from optuna.visualization._edf import _get_edf_info\n" +
            "info = _get_edf_info(studies)\n");

        AssertEdfInfo(csInfo, "info");
    }

    // ------------------------------------------------------------------ intermediate values

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 20)]
    [DataRow(42, 30)]
    public void IntermediateValues_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            t.SetConstraint("c", x);  // exercises the feasible/infeasible field.
            for (int step = 0; step < 4; step++)
                t.Report((x - 2.0) * (x - 2.0) + step, step);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        var csInfo = IntermediateValues.GetPlotInfo(csStudy);

        PyExec(
            "def _oss_obj_im(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    t.set_constraint('c', x)\n" +
            "    for step in range(4):\n" +
            "        t.report((x - 2.0) * (x - 2.0) + step, step)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_im, n_trials={nTrials})\n" +
            "from optuna.visualization._intermediate_values import _get_intermediate_plot_info\n" +
            "info = _get_intermediate_plot_info(study)\n");

        PyLong("len(info.trial_infos)").Should().Be(csInfo.TrialInfos.Count);
        for (int k = 0; k < csInfo.TrialInfos.Count; k++)
        {
            var ti = csInfo.TrialInfos[k];
            PyLong($"info.trial_infos[{k}].trial_number").Should().Be(ti.TrialNumber);
            PyBool($"info.trial_infos[{k}].feasible").Should().Be(ti.Feasible, $"trial_info {k} feasible");

            int m = (int)PyLong($"len(info.trial_infos[{k}].sorted_intermediate_values)");
            m.Should().Be(ti.SortedIntermediateValues.Count);
            for (int j = 0; j < m; j++)
            {
                PyLong($"info.trial_infos[{k}].sorted_intermediate_values[{j}][0]")
                    .Should().Be(ti.SortedIntermediateValues[j].Step);
                PyFloat($"info.trial_infos[{k}].sorted_intermediate_values[{j}][1]")
                    .Should().Be(ti.SortedIntermediateValues[j].Value);
            }
        }
    }

    // ------------------------------------------------------------------ shared assertions

    private static Study BuildQuadraticStudy(string name, int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(studyName: name, sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        return study;
    }

    private void AssertOptimizationHistoryInfo(OptimizationHistoryInfo cs, string pyVar)
    {
        PyLong($"len({pyVar}.trial_numbers)").Should().Be(cs.TrialNumbers.Count);
        for (int i = 0; i < cs.TrialNumbers.Count; i++)
            PyLong($"{pyVar}.trial_numbers[{i}]").Should().Be(cs.TrialNumbers[i]);

        AssertValuesInfo(cs.ValuesInfo, $"{pyVar}.values_info", hasStds: false);

        PyBool($"{pyVar}.best_values_info is None").Should().Be(cs.BestValuesInfo is null);
        if (cs.BestValuesInfo is not null)
            AssertValuesInfo(cs.BestValuesInfo, $"{pyVar}.best_values_info", hasStds: false);
    }

    private void AssertValuesInfo(ValuesInfo cs, string pyVar, bool hasStds)
    {
        PyStr($"{pyVar}.label_name").Should().Be(cs.LabelName);

        int n = (int)PyLong($"len({pyVar}.values)");
        n.Should().Be(cs.Values.Count);
        for (int i = 0; i < n; i++)
            PyFloat($"{pyVar}.values[{i}]").Should().Be(cs.Values[i], $"{pyVar}.values[{i}]");

        int s = (int)PyLong($"len({pyVar}.states)");
        s.Should().Be(cs.States.Count);
        for (int i = 0; i < s; i++)
            PyLong($"int({pyVar}.states[{i}].value)").Should().Be((int)cs.States[i], $"{pyVar}.states[{i}]");

        PyBool($"{pyVar}.stds is None").Should().Be(cs.Stds is null);
        if (hasStds && cs.Stds is not null)
        {
            int m = (int)PyLong($"len({pyVar}.stds)");
            m.Should().Be(cs.Stds.Count);
            for (int i = 0; i < m; i++)
                PyFloat($"{pyVar}.stds[{i}]").Should().Be(cs.Stds[i], $"{pyVar}.stds[{i}]");
        }
    }

    private void AssertEdfInfo(EdfInfo cs, string pyVar)
    {
        int nx = (int)PyLong($"len({pyVar}.x_values)");
        nx.Should().Be(cs.XValues.Length);
        for (int j = 0; j < nx; j++)
            PyFloat($"float({pyVar}.x_values[{j}])").Should().Be(cs.XValues[j], $"x_values[{j}]");

        int nl = (int)PyLong($"len({pyVar}.lines)");
        nl.Should().Be(cs.Lines.Count);
        for (int l = 0; l < nl; l++)
        {
            var line = cs.Lines[l];
            PyStr($"{pyVar}.lines[{l}].study_name").Should().Be(line.StudyName);
            int ny = (int)PyLong($"len({pyVar}.lines[{l}].y_values)");
            ny.Should().Be(line.YValues.Length);
            for (int j = 0; j < ny; j++)
                PyFloat($"float({pyVar}.lines[{l}].y_values[{j}])").Should().Be(line.YValues[j], $"line {l} y_values[{j}]");
        }
    }
}

using System;
using AwesomeAssertions;
using OptunaSharp.Distributions;
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
        PythonSession.EnsureBitExactHostOrInconclusive();

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
        PythonSession.EnsureBitExactHostOrInconclusive();

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
        PythonSession.EnsureBitExactHostOrInconclusive();

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
        PythonSession.EnsureBitExactHostOrInconclusive();

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
        PythonSession.EnsureBitExactHostOrInconclusive();

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
        PythonSession.EnsureBitExactHostOrInconclusive();

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

    // ------------------------------------------------------------------ hypervolume history

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 40)]
    [DataRow(7, 60)]
    public void HypervolumeHistory_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(MoObjective, nTrials: nTrials);
        var csInfo = HypervolumeHistory.GetInfo(csStudy, new[] { 100.0, 100.0 });

        PyExec(
            PyMoObjective +
            $"study = optuna.create_study(directions=['minimize','minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_mo, n_trials={nTrials})\n" +
            "import numpy as np\n" +
            "from optuna.visualization._hypervolume_history import _get_hypervolume_history_info\n" +
            "info = _get_hypervolume_history_info(study, np.asarray([100.0, 100.0], dtype=np.float64))\n");

        AssertHypervolumeHistory(csInfo, "info");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 50)]
    public void HypervolumeHistory_WithConstraints_MixedDirection_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Maximize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            t.SetConstraint("c", x);  // infeasible when x > 0.
            return new[] { -(x * x + y * y), (x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) };
        }, nTrials: nTrials);
        // Maximize objective 0 → reference must dominate; -(x^2+y^2) in [-50, 0], obj1 in [0, 72].
        var csInfo = HypervolumeHistory.GetInfo(csStudy, new[] { -60.0, 100.0 });

        PyExec(
            "def _oss_mo_c(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    t.set_constraint('c', x)\n" +
            "    return -(x*x + y*y), (x-1.0)*(x-1.0) + (y-1.0)*(y-1.0)\n" +
            $"study = optuna.create_study(directions=['maximize','minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_mo_c, n_trials={nTrials})\n" +
            "import numpy as np\n" +
            "from optuna.visualization._hypervolume_history import _get_hypervolume_history_info\n" +
            "info = _get_hypervolume_history_info(study, np.asarray([-60.0, 100.0], dtype=np.float64))\n");

        AssertHypervolumeHistory(csInfo, "info");
    }

    // ------------------------------------------------------------------ param importances

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void ParamImportances_SingleObjective_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(ImpObjective, nTrials: 80);
        var csInfo = ParamImportances.GetInfo(csStudy);

        PyExec(
            PyImpObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_imp, n_trials=80)\n" +
            "from optuna.visualization._param_importances import _get_importances_info\n" +
            "info = _get_importances_info(study, None, None, None, 'Objective Value')\n");

        AssertImportancesInfo(csInfo, "info");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    public void ParamImportances_MultiObjective_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize((Func<Trial, System.Collections.Generic.IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", -3, 3);
            long y = t.SuggestInt("y", 0, 10);
            return new[] { x * x + y, (x - 1.0) * (x - 1.0) - y };
        }), nTrials: 60);
        var csInfos = ParamImportances.GetInfos(csStudy);

        PyExec(
            "def _oss_mo_imp(t):\n" +
            "    x = t.suggest_float('x', -3, 3)\n" +
            "    y = t.suggest_int('y', 0, 10)\n" +
            "    return x*x + y, (x-1.0)*(x-1.0) - y\n" +
            $"study = optuna.create_study(directions=['minimize','minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_mo_imp, n_trials=60)\n" +
            "from optuna.visualization._param_importances import _get_importances_infos\n" +
            "infos = _get_importances_infos(study, None, None, None, 'Objective Value')\n");

        PyLong("len(infos)").Should().Be(csInfos.Count);
        for (int k = 0; k < csInfos.Count; k++)
            AssertImportancesInfo(csInfos[k], $"infos[{k}]");
    }

    // ------------------------------------------------------------------ slice

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 45)]
    public void Slice_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(SliceObjective, nTrials: nTrials);
        var csInfo = SlicePlot.GetPlotInfo(csStudy);

        PyExec(
            PySliceObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_slice, n_trials={nTrials})\n" +
            "from optuna.visualization._slice import _get_slice_plot_info\n" +
            "info = _get_slice_plot_info(study, None, None, 'Objective Value')\n");

        AssertSliceInfo(csInfo, "info");

        // Subset + reordered params → sorted, deduplicated.
        var csSubset = SlicePlot.GetPlotInfo(csStudy, @params: new[] { "d", "a", "a" });
        PyExec("info2 = _get_slice_plot_info(study, ['d', 'a', 'a'], None, 'Objective Value')\n");
        AssertSliceInfo(csSubset, "info2");
    }

    // ------------------------------------------------------------------ pareto front

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 40)]
    [DataRow(7, 60)]
    public void ParetoFront_TwoObjective_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            t.SetConstraint("c", x);  // some infeasible.
            return new[] { x * x + y * y, (x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) };
        }, nTrials: nTrials);
        var csInfo = ParetoFront.GetInfo(csStudy);

        PyExec(
            "def _oss_pf(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    t.set_constraint('c', x)\n" +
            "    return x*x + y*y, (x-1.0)*(x-1.0) + (y-1.0)*(y-1.0)\n" +
            $"study = optuna.create_study(directions=['minimize','minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_pf, n_trials={nTrials})\n" +
            "from optuna.visualization._pareto_front import _get_pareto_front_info\n" +
            "info = _get_pareto_front_info(study)\n");

        AssertParetoFrontInfo(csInfo, "info");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 50)]
    public void ParetoFront_ThreeObjective_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize((Func<Trial, System.Collections.Generic.IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return new[] { x * x, -(y * y), (x - y) * (x - y) };
        }), nTrials: nTrials);
        var csInfo = ParetoFront.GetInfo(csStudy);

        PyExec(
            "def _oss_pf3(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return x*x, -(y*y), (x-y)*(x-y)\n" +
            $"study = optuna.create_study(directions=['minimize','maximize','minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_pf3, n_trials={nTrials})\n" +
            "from optuna.visualization._pareto_front import _get_pareto_front_info\n" +
            "info = _get_pareto_front_info(study)\n");

        AssertParetoFrontInfo(csInfo, "info");
    }

    // ------------------------------------------------------------------ contour

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 45)]
    public void Contour_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(ContourObjective, nTrials: nTrials);
        var csInfo = Contour.GetInfo(csStudy);

        PyExec(
            PyContourObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_contour, n_trials={nTrials})\n" +
            "from optuna.visualization._contour import _get_contour_info\n" +
            "info = _get_contour_info(study, None, None, 'Objective Value')\n");

        AssertContourInfo(csInfo, "info");
    }

    // ------------------------------------------------------------------ rank

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 45)]
    public void Rank_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(ContourObjective, nTrials: nTrials);  // reuse: x/log-z/categorical-c + constraint.
        var csInfo = Rank.GetInfo(csStudy);

        PyExec(
            PyContourObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_contour, n_trials={nTrials})\n" +
            "import numpy as np\n" +
            "from optuna.visualization._rank import _get_rank_info, _get_order_with_same_order_averaging\n" +
            "info = _get_rank_info(study, None, None, 'Objective Value')\n" +
            "zs = np.array(info.zs, dtype=float)\n" +
            "cidx = (_get_order_with_same_order_averaging(zs) / (len(zs) - 1)).tolist() if len(zs) >= 2 else [0.5]\n");

        AssertRankInfo(csInfo, "info");
    }

    private void AssertRankInfo(RankPlotInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.target_name").Should().Be(cs.TargetName);
        PyBool($"{pyVar}.has_custom_target").Should().Be(cs.HasCustomTarget);

        int np = (int)PyLong($"len({pyVar}.params)");
        np.Should().Be(cs.Params.Count);
        for (int i = 0; i < np; i++)
            PyStr($"{pyVar}.params[{i}]").Should().Be(cs.Params[i]);

        int nz = (int)PyLong($"len({pyVar}.zs)");
        nz.Should().Be(cs.Zs.Length);
        for (int i = 0; i < nz; i++)
        {
            PyFloat($"float({pyVar}.zs[{i}])").Should().Be(cs.Zs[i], $"zs[{i}]");
            PyFloat($"cidx[{i}]").Should().Be(cs.ColorIdxs[i], $"color_idxs[{i}]");  // pure-numpy rank parity.
        }

        int rows = (int)PyLong($"len({pyVar}.sub_plot_infos)");
        rows.Should().Be(cs.SubPlotInfos.Count);
        for (int i = 0; i < rows; i++)
        {
            int cols = (int)PyLong($"len({pyVar}.sub_plot_infos[{i}])");
            cols.Should().Be(cs.SubPlotInfos[i].Count);
            for (int j = 0; j < cols; j++)
            {
                var sub = cs.SubPlotInfos[i][j];
                string sp = $"{pyVar}.sub_plot_infos[{i}][{j}]";
                AssertRankAxis(sub.XAxis, $"{sp}.xaxis");
                AssertRankAxis(sub.YAxis, $"{sp}.yaxis");

                int n = (int)PyLong($"len({sp}.xs)");
                n.Should().Be(sub.Xs.Count);
                for (int k = 0; k < n; k++)
                {
                    if (sub.XAxis.IsCat)
                        PyStr($"{sp}.xs[{k}]").Should().Be((string)sub.Xs[k]!, $"{sp} xs[{k}]");
                    else
                        PyFloat($"float({sp}.xs[{k}])").Should().Be(System.Convert.ToDouble(sub.Xs[k]), $"{sp} xs[{k}]");
                    if (sub.YAxis.IsCat)
                        PyStr($"{sp}.ys[{k}]").Should().Be((string)sub.Ys[k]!, $"{sp} ys[{k}]");
                    else
                        PyFloat($"float({sp}.ys[{k}])").Should().Be(System.Convert.ToDouble(sub.Ys[k]), $"{sp} ys[{k}]");

                    PyFloat($"float({sp}.zs[{k}])").Should().Be(sub.Zs[k], $"{sp} zs[{k}]");
                    PyLong($"{sp}.trials[{k}].number").Should().Be(sub.Trials[k].Number, $"{sp} trials[{k}].number");
                    PyBool($"any(v > 0.0 for v in {sp}.trials[{k}].constraints.values())")
                        .Should().Be(sub.Infeasible[k], $"{sp} infeasible[{k}]");

                    // Per-subplot color_idxs = the global color_idxs at this trial's index (self-consistent).
                    sub.ColorIdxs[k].Should().Be(cs.ColorIdxs[sub.Trials[k].Number], $"{sp} color_idxs[{k}]");
                }
            }
        }
    }

    private void AssertRankAxis(RankAxisInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.name").Should().Be(cs.Name);
        PyBool($"{pyVar}.is_log").Should().Be(cs.IsLog, $"{pyVar} is_log");
        PyBool($"{pyVar}.is_cat").Should().Be(cs.IsCat, $"{pyVar} is_cat");
        PyFloat($"{pyVar}.range[0]").Should().Be(cs.Range.Min, $"{pyVar} range[0]");
        PyFloat($"{pyVar}.range[1]").Should().Be(cs.Range.Max, $"{pyVar} range[1]");
    }

    // ------------------------------------------------------------------ parallel coordinate

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 45)]
    public void ParallelCoordinate_SingleObjective_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(ContourObjective, nTrials: nTrials);  // reuse: a/log-z... actually x/log-z/cat-c.
        var csInfo = ParallelCoordinate.GetInfo(csStudy);

        PyExec(
            PyContourObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_contour, n_trials={nTrials})\n" +
            "from optuna.visualization._parallel_coordinate import _get_parallel_coordinate_info\n" +
            "info = _get_parallel_coordinate_info(study, None, None, 'Objective Value')\n");

        AssertParallelCoordinateInfo(csInfo, "info", compareDrawOrder: true);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 40)]
    [DataRow(11, 60)]
    public void ParallelCoordinate_MissingValues_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            if (a > 0)
            {
                double d = t.SuggestFloat("d", 0, 10);  // conditional → 'd' is missing for a<=0.
                return a + d;
            }
            return a;
        }, nTrials: nTrials);
        var csInfo = ParallelCoordinate.GetInfo(csStudy);

        PyExec(
            "def _oss_pc_cond(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    if a > 0:\n" +
            "        d = t.suggest_float('d', 0, 10)\n" +
            "        return a + d\n" +
            "    return a\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_pc_cond, n_trials={nTrials})\n" +
            "from optuna.visualization._parallel_coordinate import _get_parallel_coordinate_info\n" +
            "info = _get_parallel_coordinate_info(study, None, None, 'Objective Value')\n");

        AssertParallelCoordinateInfo(csInfo, "info", compareDrawOrder: true);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 40)]
    [DataRow(7, 55)]
    public void ParallelCoordinate_MultiObjective_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(MoObjective, nTrials: nTrials);
        var csInfo = ParallelCoordinate.GetInfo(csStudy);

        PyExec(
            PyMoObjective +
            $"study = optuna.create_study(directions=['minimize','minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_mo, n_trials={nTrials})\n" +
            "from optuna.visualization._parallel_coordinate import _get_parallel_coordinate_info\n" +
            "info = _get_parallel_coordinate_info(study, None, None, 'Objective Value')\n");

        // Everything bit-exact except draw_order: it rides np.argsort over tied Pareto ranks (unstable),
        // a documented divergence. Instead verify draw_order orders the ranks non-increasingly.
        AssertParallelCoordinateInfo(csInfo, "info", compareDrawOrder: false);
        for (int i = 1; i < csInfo.DrawOrder.Count; i++)
            csInfo.ColorValues[csInfo.DrawOrder[i]].Should()
                .BeLessThanOrEqualTo(csInfo.ColorValues[csInfo.DrawOrder[i - 1]]);
    }

    // ------------------------------------------------------------------ timeline

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 20)]
    [DataRow(7, 30)]
    public void Timeline_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SetConstraint("c", x);
            t.SetUserAttr("tag", "trial-info");
            return x * x + y;
        }, nTrials: nTrials);
        var csInfo = Timeline.GetInfo(csStudy);

        PyExec(
            "def _oss_timeline(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_int('y', 0, 10)\n" +
            "    t.set_constraint('c', x)\n" +
            "    t.set_user_attr('tag', 'trial-info')\n" +
            "    return x*x + y\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_timeline, n_trials={nTrials})\n" +
            "from optuna.visualization._timeline import _get_timeline_info\n" +
            "info = _get_timeline_info(study)\n");

        PyLong("len(info.bars)").Should().Be(csInfo.Bars.Count);
        for (int k = 0; k < csInfo.Bars.Count; k++)
        {
            var bar = csInfo.Bars[k];
            PyLong($"info.bars[{k}].number").Should().Be(bar.Number);
            PyLong($"int(info.bars[{k}].state)").Should().Be((int)bar.State, $"bar {k} state");
            PyBool($"info.bars[{k}].infeasible").Should().Be(bar.Infeasible, $"bar {k} infeasible");
            // The hovertext is json.dumps({number, values, params, user_attrs}, indent=2) with <br> —
            // bit-exact (no datetimes involved), so it must match live Optuna byte-for-byte.
            PyStr($"info.bars[{k}].hovertext").Should().Be(bar.Hovertext, $"bar {k} hovertext");
            bar.Complete.Should().BeOnOrAfter(bar.Start);
        }

        // n_recent_trials slicing.
        var csRecent = Timeline.GetInfo(csStudy, nRecentTrials: 5);
        PyExec("info5 = _get_timeline_info(study, n_recent_trials=5)\n");
        PyLong("len(info5.bars)").Should().Be(csRecent.Bars.Count);
        for (int k = 0; k < csRecent.Bars.Count; k++)
            PyLong($"info5.bars[{k}].number").Should().Be(csRecent.Bars[k].Number);
    }

    private void AssertParallelCoordinateInfo(ParallelCoordinateInfo cs, string pyVar, bool compareDrawOrder)
    {
        PyStr($"{pyVar}.colorbar_title").Should().Be(cs.ColorbarTitle);
        PyBool($"{pyVar}.reverse_scale").Should().Be(cs.ReverseScale);
        PyBool($"{pyVar}.is_rank_color").Should().Be(cs.IsRankColor);

        int no = (int)PyLong($"len({pyVar}.dims_objectives)");
        no.Should().Be(cs.DimsObjectives.Count);
        for (int i = 0; i < no; i++)
            AssertDimension(cs.DimsObjectives[i], $"{pyVar}.dims_objectives[{i}]");

        int npd = (int)PyLong($"len({pyVar}.dims_params)");
        npd.Should().Be(cs.DimsParams.Count);
        for (int i = 0; i < npd; i++)
            AssertDimension(cs.DimsParams[i], $"{pyVar}.dims_params[{i}]");

        int nc = (int)PyLong($"len({pyVar}.color_values)");
        nc.Should().Be(cs.ColorValues.Count);
        for (int i = 0; i < nc; i++)
            PyFloat($"float({pyVar}.color_values[{i}])").Should().Be(cs.ColorValues[i], $"color_values[{i}]");

        PyBool($"{pyVar}.feasibility is None").Should().Be(cs.Feasibility is null);
        if (cs.Feasibility is not null)
        {
            int nf = (int)PyLong($"len({pyVar}.feasibility)");
            nf.Should().Be(cs.Feasibility.Count);
            for (int i = 0; i < nf; i++)
                PyBool($"{pyVar}.feasibility[{i}]").Should().Be(cs.Feasibility[i], $"feasibility[{i}]");
        }

        int nd = (int)PyLong($"len({pyVar}.draw_order)");
        nd.Should().Be(cs.DrawOrder.Count);
        if (compareDrawOrder)
            for (int i = 0; i < nd; i++)
                PyLong($"{pyVar}.draw_order[{i}]").Should().Be(cs.DrawOrder[i], $"draw_order[{i}]");
    }

    private void AssertDimension(DimensionInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.label").Should().Be(cs.Label, $"{pyVar} label");
        PyBool($"{pyVar}.is_log").Should().Be(cs.IsLog, $"{pyVar} is_log");
        PyBool($"{pyVar}.is_cat").Should().Be(cs.IsCat, $"{pyVar} is_cat");
        PyBool($"{pyVar}.has_missing").Should().Be(cs.HasMissing, $"{pyVar} has_missing");
        PyFloat($"float({pyVar}.range[0])").Should().Be(cs.Range.Min, $"{pyVar} range[0]");
        PyFloat($"float({pyVar}.range[1])").Should().Be(cs.Range.Max, $"{pyVar} range[1]");

        int nv = (int)PyLong($"len({pyVar}.values)");
        nv.Should().Be(cs.Values.Count, $"{pyVar} values count");
        for (int i = 0; i < nv; i++)
            PyFloat($"float({pyVar}.values[{i}])").Should().Be(cs.Values[i], $"{pyVar} values[{i}]");

        int nt = (int)PyLong($"len({pyVar}.tickvals)");
        nt.Should().Be(cs.Tickvals.Count, $"{pyVar} tickvals count");
        for (int i = 0; i < nt; i++)
            PyFloat($"float({pyVar}.tickvals[{i}])").Should().Be(cs.Tickvals[i], $"{pyVar} tickvals[{i}]");

        int ntt = (int)PyLong($"len({pyVar}.ticktext)");
        ntt.Should().Be(cs.Ticktext.Count, $"{pyVar} ticktext count");
        for (int i = 0; i < ntt; i++)
            PyStr($"{pyVar}.ticktext[{i}]").Should().Be(cs.Ticktext[i], $"{pyVar} ticktext[{i}]");
    }

    // ------------------------------------------------------------------ shared assertions

    private const string PyContourObjective =
        "def _oss_contour(t):\n" +
        "    x = t.suggest_float('x', -5, 5)\n" +
        "    z = t.suggest_float('z', 1e-3, 1e3, log=True)\n" +
        "    c = t.suggest_categorical('c', ['p', 'q', 'r'])\n" +
        "    t.set_constraint('con', x)\n" +
        "    return x*x + z/1000.0 + (1.0 if c == 'p' else 0.0)\n";

    private static double ContourObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5);
        double z = t.SuggestFloat("z", 1e-3, 1e3, log: true);
        string? c = t.SuggestCategorical("c", new CategoricalChoice[] { "p", "q", "r" }).Value as string;
        t.SetConstraint("con", x);
        return x * x + z / 1000.0 + (c == "p" ? 1.0 : 0.0);
    }

    private void AssertContourInfo(ContourInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.target_name").Should().Be(cs.TargetName);
        PyBool($"{pyVar}.reverse_scale").Should().Be(cs.ReverseScale);

        int np = (int)PyLong($"len({pyVar}.sorted_params)");
        np.Should().Be(cs.SortedParams.Count);
        for (int i = 0; i < np; i++)
            PyStr($"{pyVar}.sorted_params[{i}]").Should().Be(cs.SortedParams[i]);

        int rows = (int)PyLong($"len({pyVar}.sub_plot_infos)");
        rows.Should().Be(cs.SubPlotInfos.Count);
        for (int i = 0; i < rows; i++)
        {
            int cols = (int)PyLong($"len({pyVar}.sub_plot_infos[{i}])");
            cols.Should().Be(cs.SubPlotInfos[i].Count);
            for (int j = 0; j < cols; j++)
            {
                var sub = cs.SubPlotInfos[i][j];
                string sp = $"{pyVar}.sub_plot_infos[{i}][{j}]";
                AssertContourAxis(sub.XAxis, $"{sp}.xaxis");
                AssertContourAxis(sub.YAxis, $"{sp}.yaxis");

                PyLong($"len({sp}.z_values)").Should().Be(sub.ZValues.Count, $"{sp} z_values count");
                foreach (var kv in sub.ZValues)
                    PyFloat($"{sp}.z_values[({kv.Key.X}, {kv.Key.Y})]")
                        .Should().Be(kv.Value, $"{sp} z_values[{kv.Key.X},{kv.Key.Y}]");

                PyLong($"len({sp}.constraints)").Should().Be(sub.Constraints.Count);
                for (int k = 0; k < sub.Constraints.Count; k++)
                    PyBool($"{sp}.constraints[{k}]").Should().Be(sub.Constraints[k], $"{sp} constraints[{k}]");
            }
        }
    }

    private void AssertContourAxis(ContourAxisInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.name").Should().Be(cs.Name);
        PyBool($"{pyVar}.is_log").Should().Be(cs.IsLog, $"{pyVar} is_log");
        PyBool($"{pyVar}.is_cat").Should().Be(cs.IsCat, $"{pyVar} is_cat");
        PyFloat($"{pyVar}.range[0]").Should().Be(cs.Range.Min, $"{pyVar} range[0]");
        PyFloat($"{pyVar}.range[1]").Should().Be(cs.Range.Max, $"{pyVar} range[1]");

        int ni = (int)PyLong($"len({pyVar}.indices)");
        ni.Should().Be(cs.Indices.Count, $"{pyVar} indices count");
        for (int k = 0; k < ni; k++)
        {
            if (cs.IsCat)
                PyStr($"{pyVar}.indices[{k}]").Should().Be((string)cs.Indices[k], $"{pyVar} indices[{k}]");
            else
                PyFloat($"float({pyVar}.indices[{k}])")
                    .Should().Be(System.Convert.ToDouble(cs.Indices[k]), $"{pyVar} indices[{k}]");
        }

        int nv = (int)PyLong($"len({pyVar}.values)");
        nv.Should().Be(cs.Values.Count, $"{pyVar} values count");
        for (int k = 0; k < nv; k++)
        {
            bool isNone = PyBool($"{pyVar}.values[{k}] is None");
            isNone.Should().Be(cs.Values[k] is null, $"{pyVar} values[{k}] null");
            if (isNone)
                continue;
            if (cs.IsCat)
                PyStr($"{pyVar}.values[{k}]").Should().Be((string)cs.Values[k]!, $"{pyVar} values[{k}]");
            else
                PyFloat($"float({pyVar}.values[{k}])")
                    .Should().Be(System.Convert.ToDouble(cs.Values[k]), $"{pyVar} values[{k}]");
        }
    }

    private const string PySliceObjective =
        "def _oss_slice(t):\n" +
        "    a = t.suggest_float('a', -5, 5)\n" +
        "    b = t.suggest_float('b', 1e-3, 1e3, log=True)\n" +
        "    c = t.suggest_int('c', 0, 20)\n" +
        "    d = t.suggest_categorical('d', ['x', 'y', 'z'])\n" +
        "    t.set_constraint('con', a)\n" +
        "    return a*a + b/1000.0 + c + (1.0 if d == 'x' else 0.0)\n";

    private static double SliceObjective(Trial t)
    {
        double a = t.SuggestFloat("a", -5, 5);
        double b = t.SuggestFloat("b", 1e-3, 1e3, log: true);
        long c = t.SuggestInt("c", 0, 20);
        string? d = t.SuggestCategorical("d", new CategoricalChoice[] { "x", "y", "z" }).Value as string;
        t.SetConstraint("con", a);
        return a * a + b / 1000.0 + c + (d == "x" ? 1.0 : 0.0);
    }

    private void AssertSliceInfo(SlicePlotInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.target_name").Should().Be(cs.TargetName);

        int ns = (int)PyLong($"len({pyVar}.subplots)");
        ns.Should().Be(cs.Subplots.Count);
        for (int s = 0; s < ns; s++)
        {
            var sub = cs.Subplots[s];
            string p = $"{pyVar}.subplots[{s}]";
            PyStr($"{p}.param_name").Should().Be(sub.ParamName, $"subplot {s} param_name");
            PyBool($"{p}.is_log").Should().Be(sub.IsLog, $"subplot {s} is_log");
            PyBool($"{p}.is_numerical").Should().Be(sub.IsNumerical, $"subplot {s} is_numerical");

            int n = (int)PyLong($"len({p}.x)");
            n.Should().Be(sub.X.Count);
            PyLong($"len({p}.y)").Should().Be(sub.Y.Count);
            PyLong($"len({p}.trial_numbers)").Should().Be(sub.TrialNumbers.Count);
            PyLong($"len({p}.constraints)").Should().Be(sub.Constraints.Count);
            for (int i = 0; i < n; i++)
            {
                PyLong($"{p}.trial_numbers[{i}]").Should().Be(sub.TrialNumbers[i]);
                PyFloat($"{p}.y[{i}]").Should().Be(sub.Y[i], $"subplot {s} y[{i}]");
                PyBool($"{p}.constraints[{i}]").Should().Be(sub.Constraints[i], $"subplot {s} constraints[{i}]");
                if (sub.IsNumerical)
                    PyFloat($"float({p}.x[{i}])").Should().Be(System.Convert.ToDouble(sub.X[i]), $"subplot {s} x[{i}]");
                else
                    PyStr($"{p}.x[{i}]").Should().Be((string)sub.X[i]!, $"subplot {s} x[{i}]");
            }

            PyBool($"{p}.x_labels is None").Should().Be(sub.XLabels is null, $"subplot {s} x_labels null");
            if (sub.XLabels is not null)
            {
                int nl = (int)PyLong($"len({p}.x_labels)");
                nl.Should().Be(sub.XLabels.Count);
                for (int j = 0; j < nl; j++)
                    PyStr($"{p}.x_labels[{j}]").Should().Be((string)sub.XLabels[j].Value!, $"subplot {s} x_labels[{j}]");
            }
        }
    }

    private void AssertParetoFrontInfo(ParetoFrontInfo cs, string pyVar)
    {
        PyLong($"{pyVar}.n_targets").Should().Be(cs.NTargets);
        PyBool($"{pyVar}.include_dominated_trials").Should().Be(cs.IncludeDominatedTrials);
        PyBool($"{pyVar}.has_constraints").Should().Be(cs.HasConstraints);

        int nt = (int)PyLong($"len({pyVar}.target_names)");
        nt.Should().Be(cs.TargetNames.Count);
        for (int i = 0; i < nt; i++)
            PyStr($"{pyVar}.target_names[{i}]").Should().Be(cs.TargetNames[i]);

        int na = (int)PyLong($"len({pyVar}.axis_order)");
        na.Should().Be(cs.AxisOrder.Count);
        for (int i = 0; i < na; i++)
            PyLong($"{pyVar}.axis_order[{i}]").Should().Be(cs.AxisOrder[i]);

        AssertTrialsWithValues(cs.BestTrialsWithValues, $"{pyVar}.best_trials_with_values");
        AssertTrialsWithValues(cs.NonBestTrialsWithValues, $"{pyVar}.non_best_trials_with_values");
        AssertTrialsWithValues(cs.InfeasibleTrialsWithValues, $"{pyVar}.infeasible_trials_with_values");
    }

    private void AssertTrialsWithValues(
        System.Collections.Generic.IReadOnlyList<(FrozenTrial Trial, System.Collections.Generic.IReadOnlyList<double> Values)> cs,
        string pyList)
    {
        int n = (int)PyLong($"len({pyList})");
        n.Should().Be(cs.Count, $"{pyList} length");
        for (int k = 0; k < n; k++)
        {
            PyLong($"{pyList}[{k}][0].number").Should().Be(cs[k].Trial.Number, $"{pyList}[{k}] number");
            int m = (int)PyLong($"len({pyList}[{k}][1])");
            m.Should().Be(cs[k].Values.Count);
            for (int j = 0; j < m; j++)
                PyFloat($"{pyList}[{k}][1][{j}]").Should().Be(cs[k].Values[j], $"{pyList}[{k}][1][{j}]");
        }
    }

    private const string PyMoObjective =
        "def _oss_mo(t):\n" +
        "    x = t.suggest_float('x', -5, 5)\n" +
        "    y = t.suggest_float('y', -5, 5)\n" +
        "    return x*x + y*y, (x-1.0)*(x-1.0) + (y-1.0)*(y-1.0)\n";

    private static System.Collections.Generic.IReadOnlyList<double> MoObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5);
        double y = t.SuggestFloat("y", -5, 5);
        return new[] { x * x + y * y, (x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) };
    }

    private const string PyImpObjective =
        "def _oss_imp(t):\n" +
        "    x1 = t.suggest_float('x1', -5, 5)\n" +
        "    x2 = t.suggest_int('x2', 0, 20)\n" +
        "    x3 = t.suggest_categorical('x3', ['a', 'b', 'c'])\n" +
        "    return x1*x1 + x2 + (1.0 if x3 == 'a' else 0.0)\n";

    private static double ImpObjective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", -5, 5);
        long x2 = t.SuggestInt("x2", 0, 20);
        string? x3 = t.SuggestCategorical("x3", new CategoricalChoice[] { "a", "b", "c" }).Value as string;
        return x1 * x1 + x2 + (x3 == "a" ? 1.0 : 0.0);
    }

    private void AssertHypervolumeHistory(HypervolumeHistoryInfo cs, string pyVar)
    {
        int n = (int)PyLong($"len({pyVar}.trial_numbers)");
        n.Should().Be(cs.TrialNumbers.Count);
        for (int i = 0; i < n; i++)
            PyLong($"{pyVar}.trial_numbers[{i}]").Should().Be(cs.TrialNumbers[i]);

        int m = (int)PyLong($"len({pyVar}.values)");
        m.Should().Be(cs.Values.Count);
        // Two objectives → compute_hypervolume rides the dot-free 2-D path, so it is byte-exact.
        for (int i = 0; i < m; i++)
            PyFloat($"float({pyVar}.values[{i}])").Should().Be(cs.Values[i], $"{pyVar}.values[{i}]");
    }

    // Importance values inherit PED-ANOVA's erf/ucrt few-ULP tolerance; names/order/labels are exact.
    private const double ImportanceTolerance = 1e-10;

    private void AssertImportancesInfo(ImportancesInfo cs, string pyVar)
    {
        PyStr($"{pyVar}.target_name").Should().Be(cs.TargetName);

        int n = (int)PyLong($"len({pyVar}.param_names)");
        n.Should().Be(cs.ParamNames.Count);
        for (int i = 0; i < n; i++)
        {
            PyStr($"{pyVar}.param_names[{i}]").Should().Be(cs.ParamNames[i]);
            PyFloat($"{pyVar}.importance_values[{i}]")
                .Should().BeApproximately(cs.ImportanceValues[i], ImportanceTolerance, $"{pyVar}.importance_values[{i}]");
            PyStr($"{pyVar}.importance_labels[{i}]").Should().Be(cs.ImportanceLabels[i], $"{pyVar}.importance_labels[{i}]");
        }
    }

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

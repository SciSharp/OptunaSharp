using System;
using Python.Runtime;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     Selects the BoTorch acquisition function <see cref="BoTorchSampler" /> optimizes — the .NET,
///     <see cref="PyObject" />-free analog of Optuna's <c>candidates_func</c> argument. The default,
///     <see cref="Auto" />, chooses automatically from the number of objectives, whether a constraint is
///     specified, and whether running trials are considered (Optuna's <c>candidates_func=None</c> /
///     <c>_get_default_candidates_func</c>).
/// </summary>
public enum BoTorchCandidatesFunc
{
    /// <summary>
    ///     Choose automatically (Optuna's <c>candidates_func=None</c> / <c>_get_default_candidates_func</c>):
    ///     <c>n_obj&gt;3 &amp;&amp; !constrained &amp;&amp; !running → </c><see cref="Ehvi" />;
    ///     <c>n_obj&gt;3 → </c><see cref="QParEgo" />; <c>n_obj&gt;1 → </c><see cref="Qehvi" />;
    ///     <c>running → </c><see cref="Qei" />; else <see cref="LogEi" />.
    /// </summary>
    Auto = 0,

    /// <summary>Log Expected Improvement (LogEI) — the single-objective default.</summary>
    LogEi,

    /// <summary>Quasi-MC batch Expected Improvement (qEI).</summary>
    Qei,

    /// <summary>Quasi-MC batch Log Expected Improvement (qLogEI) — the numerically stable qEI.</summary>
    QLogEi,

    /// <summary>Quasi-MC batch Noisy Expected Improvement (qNEI) — for noisy objectives.</summary>
    QNei,

    /// <summary>Quasi-MC batch Expected Hypervolume Improvement (qEHVI) — the multi-objective default (≤3 objectives).</summary>
    Qehvi,

    /// <summary>Expected Hypervolume Improvement (EHVI) — analytic; the multi-objective default when &gt;3 objectives and unconstrained.</summary>
    Ehvi,

    /// <summary>Quasi-MC batch Noisy Expected Hypervolume Improvement (qNEHVI) — for noisy multi-objective problems.</summary>
    Qnehvi,

    /// <summary>Quasi-MC extended ParEGO (qParEGO) — the multi-objective default when &gt;3 objectives (or constrained).</summary>
    QParEgo,

    /// <summary>Quasi-MC extended ParEGO backed by qLogEI (qLogEI-ParEGO) — the numerically stable qParEGO.</summary>
    QLogEiParEgo,

    /// <summary>Quasi-MC batch Knowledge Gradient (qKG) — single-objective; never an <see cref="Auto" /> default.</summary>
    Qkg,

    /// <summary>Quasi-MC batch Hypervolume Knowledge Gradient (qHVKG) — multi-objective; never an <see cref="Auto" /> default.</summary>
    Qhvkg,
}

/// <summary>
///     The native <c>candidates_func</c> contract — takes the training data, objectives, constraints,
///     search-space bounds and pending points as <c>torch.Tensor</c>s and returns the next candidates as
///     a <c>torch.Tensor</c>. <b>Internal:</b> it operates on <see cref="PyObject" />s (torch tensors),
///     which the bridge keeps off its public surface — callers select an acquisition with the
///     <see cref="BoTorchCandidatesFunc" /> enum instead.
/// </summary>
internal delegate PyObject NativeCandidatesFunc(
    PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX);

/// <summary>
///     The built-in <c>candidates_func</c> implementations — faithful ports of the functions in
///     <c>optuna_integration/botorch/botorch.py</c>, each expressed as a near-verbatim BoTorch snippet
///     run through the Pythonic <see cref="Python.botorch" /> layer (bind the tensors, fit the
///     <c>SingleTaskGP</c>/<c>ModelListGP</c>, build the acquisition, optimize it — all in real BoTorch).
///     Internal by design (they speak <see cref="PyObject" />); the sampler resolves the public
///     <see cref="BoTorchCandidatesFunc" /> enum to one of these. Every method must be called under the
///     GIL.
///     <para>Each body carries both the unconstrained and the constrained branch of its Python twin;
///     the pre-numeric guards (single-objective checks, EHVI's constraint rejection) stay in C# so they
///     surface as typed exceptions with verbatim messages, while the version-requirement ImportErrors and
///     the <c>_get_constraint_funcs</c> / <c>_validate_botorch_version_for_constrained_opt</c> helpers
///     live in the ported snippet (see <see cref="Python.botorch" />).</para>
/// </summary>
internal static class BoTorchCandidates
{
    /// <summary>
    ///     Map a <see cref="BoTorchCandidatesFunc" /> selection to a concrete implementation — port of
    ///     <c>_get_default_candidates_func</c> for <see cref="BoTorchCandidatesFunc.Auto" />, an explicit
    ///     lookup otherwise.
    /// </summary>
    internal static NativeCandidatesFunc Resolve(
        BoTorchCandidatesFunc selection, int nObjectives, bool hasConstraint, bool considerRunningTrials)
    {
        if (selection != BoTorchCandidatesFunc.Auto)
            return selection switch
            {
                BoTorchCandidatesFunc.LogEi => LogEi,
                BoTorchCandidatesFunc.Qei => Qei,
                BoTorchCandidatesFunc.QLogEi => QLogEi,
                BoTorchCandidatesFunc.QNei => QNei,
                BoTorchCandidatesFunc.Qehvi => Qehvi,
                BoTorchCandidatesFunc.Ehvi => Ehvi,
                BoTorchCandidatesFunc.Qnehvi => Qnehvi,
                BoTorchCandidatesFunc.QParEgo => QParEgo,
                BoTorchCandidatesFunc.QLogEiParEgo => QLogEiParEgo,
                BoTorchCandidatesFunc.Qkg => Qkg,
                BoTorchCandidatesFunc.Qhvkg => Qhvkg,
                _ => throw new NotSupportedException($"Unknown candidates function '{selection}'."),
            };

        // Auto == _get_default_candidates_func (botorch.py:1029).
        if (nObjectives > 3 && !hasConstraint && !considerRunningTrials)
            return Ehvi;
        if (nObjectives > 3)
            return QParEgo;
        if (nObjectives > 1)
            return Qehvi;
        return considerRunningTrials ? Qei : LogEi;
    }

    // ---- single-objective ------------------------------------------------------------------------

    /// <summary>Log Expected Improvement (LogEI). Port of <c>logei_candidates_func</c>.</summary>
    internal static PyObject LogEi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "logEI");
        return Python.botorch.evalStmts(LogEiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    /// <summary>Quasi-MC batch Expected Improvement (qEI). Port of <c>qei_candidates_func</c>.</summary>
    internal static PyObject Qei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qEI");
        return Python.botorch.evalStmts(QeiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    /// <summary>Quasi-MC batch Log Expected Improvement (qLogEI). Port of <c>qlogei_candidates_func</c>.</summary>
    internal static PyObject QLogEi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qLogEI");
        return Python.botorch.evalStmts(QLogEiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    /// <summary>Quasi-MC batch Noisy Expected Improvement (qNEI). Port of <c>qnei_candidates_func</c>.</summary>
    internal static PyObject QNei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qNEI");
        return Python.botorch.evalStmts(QNeiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    // ---- multi-objective -------------------------------------------------------------------------

    /// <summary>Quasi-MC batch Expected Hypervolume Improvement (qEHVI). Port of <c>qehvi_candidates_func</c>.</summary>
    internal static PyObject Qehvi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
        => Python.botorch.evalStmts(QehviBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));

    /// <summary>Expected Hypervolume Improvement (EHVI). Port of <c>ehvi_candidates_func</c>.</summary>
    internal static PyObject Ehvi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        if (trainCon is not null)
            throw new NumSharp.ValueError("Constraints are not supported with ehvi_candidates_func.");
        return Python.botorch.evalStmts(EhviBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("bounds", bounds));
    }

    /// <summary>Quasi-MC batch Noisy Expected Hypervolume Improvement (qNEHVI). Port of <c>qnehvi_candidates_func</c>.</summary>
    internal static PyObject Qnehvi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
        => Python.botorch.evalStmts(QnehviBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));

    /// <summary>Quasi-MC extended ParEGO (qParEGO). Port of <c>qparego_candidates_func</c>.</summary>
    internal static PyObject QParEgo(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
        => Python.botorch.evalStmts(QParEgoBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));

    /// <summary>Quasi-MC extended ParEGO backed by qLogEI (qLogEI-ParEGO). Port of <c>qlogei_parego_candidates_func</c>.</summary>
    internal static PyObject QLogEiParEgo(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
        => Python.botorch.evalStmts(QLogEiParEgoBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));

    // ---- knowledge gradient ----------------------------------------------------------------------

    /// <summary>Quasi-MC batch Knowledge Gradient (qKG). Port of <c>qkg_candidates_func</c>.</summary>
    internal static PyObject Qkg(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qKG");
        return Python.botorch.evalStmts(QkgBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    /// <summary>Quasi-MC batch Hypervolume Knowledge Gradient (qHVKG). Port of <c>qhvkg_candidates_func</c>.</summary>
    internal static PyObject Qhvkg(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
        => Python.botorch.evalStmts(QhvkgBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("train_con", trainCon ?? PyObject.None),
            ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));

    // ---- the ported candidates_func bodies (near-verbatim botorch.py) ----------------------------

    private const string LogEiBody =
        "if not _logei_ok:\n" +
        "    raise ImportError('logei_candidates_func requires botorch >=0.8.1. Please upgrade botorch or use qei_candidates_func as candidates_func instead.')\n" +
        "n_constraints = train_con.size(1) if train_con is not None else 0\n" +
        "if n_constraints > 0:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    is_feas = (train_con <= 0).all(dim=-1)\n" +
        "    train_obj_feas = train_obj[is_feas]\n" +
        "    if train_obj_feas.numel() == 0:\n" +
        "        best_f = train_obj.min()\n" +
        "    else:\n" +
        "        best_f = train_obj_feas.max()\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    best_f = train_obj.max()\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "if n_constraints > 0:\n" +
        "    acqf = LogConstrainedExpectedImprovement(model=model, best_f=best_f, objective_index=0, constraints={i: (None, 0.0) for i in range(1, n_constraints + 1)})\n" +
        "else:\n" +
        "    acqf = LogExpectedImprovement(model=model, best_f=best_f)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=512, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QeiBody =
        "if train_con is not None:\n" +
        "    _validate_botorch_version_for_constrained_opt('qei_candidates_func')\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    is_feas = (train_con <= 0).all(dim=-1)\n" +
        "    train_obj_feas = train_obj[is_feas]\n" +
        "    if train_obj_feas.numel() == 0:\n" +
        "        best_f = torch.zeros(())\n" +
        "    else:\n" +
        "        best_f = train_obj_feas.max()\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    additonal_qei_kwargs = {'objective': GenericMCObjective(lambda Z, X: Z[..., 0]), 'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    best_f = train_obj.max()\n" +
        "    additonal_qei_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "acqf = qExpectedImprovement(model=model, best_f=best_f, sampler=_get_sobol_qmc_normal_sampler(256), X_pending=pending_x, **additonal_qei_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=512, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QLogEiBody =
        "if not _qlogei_ok:\n" +
        "    raise ImportError('qlogei_candidates_func requires botorch>=0.9.0. Please upgrade botorch or use qei_candidates_func as candidates_func instead.')\n" +
        "if train_con is not None:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    is_feas = (train_con <= 0).all(dim=-1)\n" +
        "    train_obj_feas = train_obj[is_feas]\n" +
        "    if train_obj_feas.numel() == 0:\n" +
        "        best_f = torch.zeros(())\n" +
        "    else:\n" +
        "        best_f = train_obj_feas.max()\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    additional_qlogei_kwargs = {'objective': GenericMCObjective(lambda Z, X: Z[..., 0]), 'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    best_f = train_obj.max()\n" +
        "    additional_qlogei_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "acqf = qLogExpectedImprovement(model=model, best_f=best_f, sampler=_get_sobol_qmc_normal_sampler(256), X_pending=pending_x, **additional_qlogei_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=512, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QNeiBody =
        "if train_con is not None:\n" +
        "    _validate_botorch_version_for_constrained_opt('qnei_candidates_func')\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    additional_qnei_kwargs = {'objective': GenericMCObjective(lambda Z, X: Z[..., 0]), 'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    additional_qnei_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "acqf = qNoisyExpectedImprovement(model=model, X_baseline=train_x, sampler=_get_sobol_qmc_normal_sampler(256), X_pending=pending_x, **additional_qnei_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=512, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QehviBody =
        "n_objectives = train_obj.size(-1)\n" +
        "if train_con is not None:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    is_feas = (train_con <= 0).all(dim=-1)\n" +
        "    train_obj_feas = train_obj[is_feas]\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    additional_qehvi_kwargs = {'objective': IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))), 'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    train_obj_feas = train_obj\n" +
        "    additional_qehvi_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "if n_objectives > 4:\n" +
        "    alpha = 10 ** (-8 + n_objectives)\n" +
        "else:\n" +
        "    alpha = 0.0\n" +
        "ref_point = train_obj.min(dim=0).values - 1e-8\n" +
        "partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj_feas, alpha=alpha)\n" +
        "ref_point_list = ref_point.tolist()\n" +
        "if _qloghvi_ok and partitioning.get_hypercell_bounds().shape[-2] > 0:\n" +
        "    hypervol_improvement_method = qLogExpectedHypervolumeImprovement\n" +
        "else:\n" +
        "    hypervol_improvement_method = monte_carlo.qExpectedHypervolumeImprovement\n" +
        "acqf = hypervol_improvement_method(model=model, ref_point=ref_point_list, partitioning=partitioning, sampler=_get_sobol_qmc_normal_sampler(256), X_pending=pending_x, **additional_qehvi_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=20, raw_samples=1024, options={'batch_limit': 5, 'maxiter': 200, 'nonnegative': True}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string EhviBody =
        "n_objectives = train_obj.size(-1)\n" +
        "train_y = train_obj\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "if n_objectives > 4:\n" +
        "    alpha = 10 ** (-8 + n_objectives)\n" +
        "else:\n" +
        "    alpha = 0.0\n" +
        "ref_point = train_obj.min(dim=0).values - 1e-8\n" +
        "partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y, alpha=alpha)\n" +
        "ref_point_list = ref_point.tolist()\n" +
        "acqf = ExpectedHypervolumeImprovement(model=model, ref_point=ref_point_list, partitioning=partitioning)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=20, raw_samples=1024, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QnehviBody =
        "n_objectives = train_obj.size(-1)\n" +
        "if train_con is not None:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    additional_qnehvi_kwargs = {'objective': IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))), 'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    additional_qnehvi_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "if n_objectives > 4:\n" +
        "    alpha = 10 ** (-8 + n_objectives)\n" +
        "else:\n" +
        "    alpha = 0.0\n" +
        "ref_point = train_obj.min(dim=0).values - 1e-8\n" +
        "ref_point_list = ref_point.tolist()\n" +
        "if _qloghvi_ok:\n" +
        "    noisy_hypervol_improvement_method = qLogNoisyExpectedHypervolumeImprovement\n" +
        "else:\n" +
        "    noisy_hypervol_improvement_method = monte_carlo.qNoisyExpectedHypervolumeImprovement\n" +
        "acqf = noisy_hypervol_improvement_method(model=model, ref_point=ref_point_list, X_baseline=train_x, alpha=alpha, prune_baseline=True, sampler=_get_sobol_qmc_normal_sampler(256), X_pending=pending_x, **additional_qnehvi_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=20, raw_samples=1024, options={'batch_limit': 5, 'maxiter': 200, 'nonnegative': True}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QParEgoBody =
        "n_objectives = train_obj.size(-1)\n" +
        "weights = sample_simplex(n_objectives).squeeze()\n" +
        "scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)\n" +
        "if train_con is not None:\n" +
        "    _validate_botorch_version_for_constrained_opt('qparego_candidates_func')\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    objective = GenericMCObjective(lambda Z, X: scalarization(Z[..., :n_objectives]))\n" +
        "    additional_qei_kwargs = {'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    objective = GenericMCObjective(scalarization)\n" +
        "    additional_qei_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "acqf = qExpectedImprovement(model=model, best_f=objective(train_y).max(), sampler=_get_sobol_qmc_normal_sampler(256), objective=objective, X_pending=pending_x, **additional_qei_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=20, raw_samples=1024, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QLogEiParEgoBody =
        "if not _qlogei_ok:\n" +
        "    raise ImportError('qlogei_parego_candidates_func requires botorch>=0.9.0. Please upgrade botorch or use qparego_candidates_func as candidates_func instead.')\n" +
        "n_objectives = train_obj.size(-1)\n" +
        "weights = sample_simplex(n_objectives).squeeze()\n" +
        "scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)\n" +
        "if train_con is not None:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    objective = GenericMCObjective(lambda Z, X: scalarization(Z[..., :n_objectives]))\n" +
        "    additional_qlogei_kwargs = {'constraints': _get_constraint_funcs(n_constraints)}\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    objective = GenericMCObjective(scalarization)\n" +
        "    additional_qlogei_kwargs = {}\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "acqf = qLogExpectedImprovement(model=model, best_f=objective(train_y).max(), sampler=_get_sobol_qmc_normal_sampler(256), objective=objective, X_pending=pending_x, **additional_qlogei_kwargs)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=20, raw_samples=1024, options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QkgBody =
        "if train_con is not None:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "    n_constraints = train_con.size(1)\n" +
        "    objective = ConstrainedMCObjective(objective=lambda Z, X: Z[..., 0], constraints=_get_constraint_funcs(n_constraints))\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "    objective = None\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "mll = ExactMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "acqf = qKnowledgeGradient(model=model, num_fantasies=256, objective=objective, X_pending=pending_x)\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=512, options={'batch_limit': 8, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string QhvkgBody =
        "if not _qhvkg_ok:\n" +
        "    raise ImportError('qhvkg_candidates_func requires botorch >=0.9.5. Please upgrade botorch or use qehvi_candidates_func as candidates_func instead.')\n" +
        "if train_con is not None:\n" +
        "    train_y = torch.cat([train_obj, train_con], dim=-1)\n" +
        "else:\n" +
        "    train_y = train_obj\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "models = [SingleTaskGP(train_x, train_y[..., [i]], outcome_transform=Standardize(m=1)) for i in range(train_y.shape[-1])]\n" +
        "model = ModelListGP(*models)\n" +
        "mll = SumMarginalLogLikelihood(model.likelihood, model)\n" +
        "fit_gpytorch_mll(mll)\n" +
        "n_constraints = train_con.size(1) if train_con is not None else 0\n" +
        "objective = multi_objective_objective.FeasibilityWeightedMCMultiOutputObjective(model, X_baseline=train_x, constraint_idcs=[-n_constraints + i for i in range(n_constraints)])\n" +
        "ref_point = train_obj.min(dim=0).values - 1e-8\n" +
        "acqf = hypervolume_knowledge_gradient.qHypervolumeKnowledgeGradient(model=model, ref_point=ref_point, num_fantasies=16, X_pending=pending_x, objective=objective, sampler=ListSampler(*[SobolQMCNormalSampler(sample_shape=torch.Size([16])) for _ in range(model.num_outputs)]), inner_sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])))\n" +
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=1, raw_samples=1024, options={'batch_limit': 4, 'maxiter': 200, 'nonnegative': True}, sequential=False)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    // ---- guards ---------------------------------------------------------------------------------

    private static void RequireSingleObjective(PyObject trainObj, string name)
    {
        if (Python.torch.Tensor.size_t(trainObj, -1) != 1)
            throw new NumSharp.ValueError($"Objective may only contain single values with {name}.");
    }
}

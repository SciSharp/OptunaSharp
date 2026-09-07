using System;
using Python.Runtime;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     Selects the BoTorch acquisition function <see cref="BoTorchSampler" /> optimizes — the .NET,
///     <see cref="PyObject" />-free analog of Optuna's <c>candidates_func</c> argument. The default,
///     <see cref="Auto" />, chooses automatically from the number of objectives and whether running
///     trials are considered (Optuna's <c>candidates_func=None</c>).
///     <para><b>Phase A</b> exposes the single-objective functions. Multi-objective, constrained, and
///     knowledge-gradient acquisitions arrive with later phases (new enum members).</para>
/// </summary>
public enum BoTorchCandidatesFunc
{
    /// <summary>Choose automatically (Optuna's <c>candidates_func=None</c>): single-objective → <see cref="LogEi" />, or <see cref="Qei" /> when running trials are considered.</summary>
    Auto = 0,

    /// <summary>Log Expected Improvement (LogEI) — the single-objective default.</summary>
    LogEi,

    /// <summary>Quasi-MC batch Expected Improvement (qEI).</summary>
    Qei,

    /// <summary>Quasi-MC batch Log Expected Improvement (qLogEI) — the numerically stable qEI.</summary>
    QLogEi,

    /// <summary>Quasi-MC batch Noisy Expected Improvement (qNEI) — for noisy objectives.</summary>
    QNei,
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
///     <c>SingleTaskGP</c>, build the acquisition, optimize it — all in real BoTorch). Internal by design
///     (they speak <see cref="PyObject" />); the sampler resolves the public
///     <see cref="BoTorchCandidatesFunc" /> enum to one of these. Every method must be called under the
///     GIL.
///     <para><b>Phase A</b> ships the single-objective path; multi-objective / constrained /
///     knowledge-gradient variants follow in later phases (the constrained branch of each raises for
///     now).</para>
/// </summary>
internal static class BoTorchCandidates
{
    /// <summary>
    ///     Map a <see cref="BoTorchCandidatesFunc" /> selection to a concrete implementation — port of
    ///     <c>_get_default_candidates_func</c> for <see cref="BoTorchCandidatesFunc.Auto" />, an explicit
    ///     lookup otherwise. Phase A: single-objective only (multi-objective auto-selection raises until
    ///     Phase B).
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
                _ => throw new NotSupportedException($"Unknown candidates function '{selection}'."),
            };

        // Auto == _get_default_candidates_func. The full Python selector (Phase B/C completes the
        // multi-objective branches): n>3 && !con && !running -> ehvi; n>3 -> qparego; n>1 -> qehvi;
        // running -> qei; else logei.
        if (nObjectives > 1)
            throw new NotSupportedException(
                "Multi-objective BoTorchSampler (qEHVI/EHVI/qParEGO/qNEHVI) is not ported yet (Phase B). " +
                "Use a single-objective study.");
        return considerRunningTrials ? Qei : LogEi;
    }

    /// <summary>Log Expected Improvement (LogEI). Port of <c>logei_candidates_func</c>.</summary>
    internal static PyObject LogEi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "logEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("logei_candidates_func");

        return Python.botorch.evalStmts(LogEiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("bounds", bounds));
    }

    /// <summary>Quasi-MC batch Expected Improvement (qEI). Port of <c>qei_candidates_func</c>.</summary>
    internal static PyObject Qei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qei_candidates_func");

        return Python.botorch.evalStmts(QeiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    /// <summary>Quasi-MC batch Log Expected Improvement (qLogEI). Port of <c>qlogei_candidates_func</c>.</summary>
    internal static PyObject QLogEi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qLogEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qlogei_candidates_func");

        return Python.botorch.evalStmts(QLogEiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    /// <summary>Quasi-MC batch Noisy Expected Improvement (qNEI). Port of <c>qnei_candidates_func</c>.</summary>
    internal static PyObject QNei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qNEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qnei_candidates_func");

        return Python.botorch.evalStmts(QNeiBody, "result",
            ("train_x", trainX), ("train_obj", trainObj), ("bounds", bounds), ("pending_x", pendingX ?? PyObject.None));
    }

    // ---- the ported candidates_func bodies (near-verbatim botorch.py, single-objective) ----------

    private const string StandardBoundsAndOptimize =
        "standard_bounds = torch.zeros_like(bounds)\n" +
        "standard_bounds[1] = 1\n" +
        "candidates, _ = optimize_acqf(\n" +
        "    acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=512,\n" +
        "    options={'batch_limit': 5, 'maxiter': 200}, sequential=True)\n" +
        "result = unnormalize(candidates.detach(), bounds=bounds)\n";

    private const string LogEiBody =
        "train_y = train_obj\n" +
        "best_f = train_obj.max()\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))\n" +
        "acqf = LogExpectedImprovement(model=model, best_f=best_f)\n" +
        StandardBoundsAndOptimize;

    private const string QeiBody =
        "train_y = train_obj\n" +
        "best_f = train_obj.max()\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))\n" +
        "acqf = qExpectedImprovement(model=model, best_f=best_f, sampler=SobolQMCNormalSampler(torch.Size([256])), X_pending=pending_x)\n" +
        StandardBoundsAndOptimize;

    private const string QLogEiBody =
        "train_y = train_obj\n" +
        "best_f = train_obj.max()\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))\n" +
        "acqf = qLogExpectedImprovement(model=model, best_f=best_f, sampler=SobolQMCNormalSampler(torch.Size([256])), X_pending=pending_x)\n" +
        StandardBoundsAndOptimize;

    private const string QNeiBody =
        "train_y = train_obj\n" +
        "train_x = normalize(train_x, bounds=bounds)\n" +
        "if pending_x is not None:\n" +
        "    pending_x = normalize(pending_x, bounds=bounds)\n" +
        "model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))\n" +
        "fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))\n" +
        "acqf = qNoisyExpectedImprovement(model=model, X_baseline=train_x, sampler=SobolQMCNormalSampler(torch.Size([256])), X_pending=pending_x)\n" +
        StandardBoundsAndOptimize;

    // ---- guards ---------------------------------------------------------------------------------

    private static void RequireSingleObjective(PyObject trainObj, string name)
    {
        using PyObject dim = (-1).ToPython();
        using PyObject size = trainObj.InvokeMethod("size", dim);
        if (size.As<long>() != 1)
            throw new NumSharp.ValueError($"Objective may only contain single values with {name}.");
    }

    private static NotSupportedException ConstraintsNotYetSupported(string func)
        => new($"Constrained optimization in {func} is not ported yet (Phase C). " +
               "Use an unconstrained single-objective study for now.");
}

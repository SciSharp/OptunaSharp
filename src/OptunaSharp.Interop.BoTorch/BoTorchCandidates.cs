using System;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

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
///     The built-in <c>candidates_func</c> implementations — faithful C# ports of the functions in
///     <c>optuna_integration/botorch/botorch.py</c>, each orchestrating the <b>real BoTorch</b>
///     (<c>SingleTaskGP</c> → <c>fit_gpytorch_mll</c> → an acquisition function → <c>optimize_acqf</c>)
///     over <c>torch.Tensor</c>s via pythonnet. Internal by design (they speak <see cref="PyObject" />);
///     the sampler resolves the public <see cref="BoTorchCandidatesFunc" /> enum to one of these. Every
///     method must be called under the GIL.
///     <para><b>Phase A</b> ships the single-objective path; multi-objective / constrained /
///     knowledge-gradient variants follow in later phases. <b>Cached-module discipline:</b>
///     <see cref="PythonRuntime.Import" /> returns per-process cached modules — used as plain locals,
///     never disposed; only fresh <c>GetAttr</c>/<c>Invoke</c> results are wrapped in <c>using</c>.</para>
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

        PyObject trainY = trainObj;
        using PyObject bestF = trainObj.InvokeMethod("max");
        using PyObject normX = Normalize(trainX, bounds);
        using PyObject model = FitSingleTaskGp(normX, trainY);

        PyObject analytic = PythonRuntime.Import("botorch.acquisition.analytic");
        using PyObject logEiCtor = analytic.GetAttr("LogExpectedImprovement");
        using var acqfKwargs = new PyDict();
        acqfKwargs["model"] = model;
        acqfKwargs["best_f"] = bestF;
        using PyObject acqf = logEiCtor.Invoke(Array.Empty<PyObject>(), acqfKwargs);

        return OptimizeAndUnnormalize(acqf, bounds, numRestarts: 10, rawSamples: 512, nonnegative: false);
    }

    /// <summary>Quasi-MC batch Expected Improvement (qEI). Port of <c>qei_candidates_func</c>.</summary>
    internal static PyObject Qei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qei_candidates_func");

        PyObject trainY = trainObj;
        using PyObject bestF = trainObj.InvokeMethod("max");
        using PyObject normX = Normalize(trainX, bounds);
        using PyObject? normPending = MaybeNormalize(pendingX, bounds);
        using PyObject model = FitSingleTaskGp(normX, trainY);

        PyObject mc = PythonRuntime.Import("botorch.acquisition.monte_carlo");
        using PyObject qeiCtor = mc.GetAttr("qExpectedImprovement");
        using PyObject sampler = SobolQmcNormalSampler(256);
        using var acqfKwargs = new PyDict();
        acqfKwargs["model"] = model;
        acqfKwargs["best_f"] = bestF;
        acqfKwargs["sampler"] = sampler;
        acqfKwargs["X_pending"] = XPending(normPending);
        using PyObject acqf = qeiCtor.Invoke(Array.Empty<PyObject>(), acqfKwargs);

        return OptimizeAndUnnormalize(acqf, bounds, numRestarts: 10, rawSamples: 512, nonnegative: false);
    }

    /// <summary>Quasi-MC batch Log Expected Improvement (qLogEI). Port of <c>qlogei_candidates_func</c>.</summary>
    internal static PyObject QLogEi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qLogEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qlogei_candidates_func");

        PyObject trainY = trainObj;
        using PyObject bestF = trainObj.InvokeMethod("max");
        using PyObject normX = Normalize(trainX, bounds);
        using PyObject? normPending = MaybeNormalize(pendingX, bounds);
        using PyObject model = FitSingleTaskGp(normX, trainY);

        PyObject logei = PythonRuntime.Import("botorch.acquisition.logei");
        using PyObject qLogEiCtor = logei.GetAttr("qLogExpectedImprovement");
        using PyObject sampler = SobolQmcNormalSampler(256);
        using var acqfKwargs = new PyDict();
        acqfKwargs["model"] = model;
        acqfKwargs["best_f"] = bestF;
        acqfKwargs["sampler"] = sampler;
        acqfKwargs["X_pending"] = XPending(normPending);
        using PyObject acqf = qLogEiCtor.Invoke(Array.Empty<PyObject>(), acqfKwargs);

        return OptimizeAndUnnormalize(acqf, bounds, numRestarts: 10, rawSamples: 512, nonnegative: false);
    }

    /// <summary>Quasi-MC batch Noisy Expected Improvement (qNEI). Port of <c>qnei_candidates_func</c>.</summary>
    internal static PyObject QNei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qNEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qnei_candidates_func");

        PyObject trainY = trainObj;
        using PyObject normX = Normalize(trainX, bounds);
        using PyObject? normPending = MaybeNormalize(pendingX, bounds);
        using PyObject model = FitSingleTaskGp(normX, trainY);

        PyObject mc = PythonRuntime.Import("botorch.acquisition.monte_carlo");
        using PyObject qNeiCtor = mc.GetAttr("qNoisyExpectedImprovement");
        using PyObject sampler = SobolQmcNormalSampler(256);
        using var acqfKwargs = new PyDict();
        acqfKwargs["model"] = model;
        acqfKwargs["X_baseline"] = normX;
        acqfKwargs["sampler"] = sampler;
        acqfKwargs["X_pending"] = XPending(normPending);
        using PyObject acqf = qNeiCtor.Invoke(Array.Empty<PyObject>(), acqfKwargs);

        return OptimizeAndUnnormalize(acqf, bounds, numRestarts: 10, rawSamples: 512, nonnegative: false);
    }

    // ---- shared botorch orchestration -----------------------------------------------------------

    /// <summary><c>normalize(x, bounds=bounds)</c> from <c>botorch.utils.transforms</c>.</summary>
    private static PyObject Normalize(PyObject x, PyObject bounds)
    {
        PyObject transforms = PythonRuntime.Import("botorch.utils.transforms");
        using PyObject normalize = transforms.GetAttr("normalize");
        using var kwargs = new PyDict();
        kwargs["bounds"] = bounds;
        return normalize.Invoke(new[] { x }, kwargs);
    }

    private static PyObject? MaybeNormalize(PyObject? x, PyObject bounds)
        => x is null ? null : Normalize(x, bounds);

    private static PyObject XPending(PyObject? normalizedPending)
        => normalizedPending ?? PyObject.None;

    /// <summary>
    ///     <c>model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)));
    ///     fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))</c> — the fitted GP.
    /// </summary>
    private static PyObject FitSingleTaskGp(PyObject trainXNormalized, PyObject trainY)
    {
        long m = Size(trainY, -1);

        PyObject outcome = PythonRuntime.Import("botorch.models.transforms.outcome");
        using PyObject standardizeCtor = outcome.GetAttr("Standardize");
        using var stdKwargs = new PyDict();
        stdKwargs["m"] = new PyInt(m);
        using PyObject standardize = standardizeCtor.Invoke(Array.Empty<PyObject>(), stdKwargs);

        PyObject models = PythonRuntime.Import("botorch.models");
        using PyObject singleTaskGpCtor = models.GetAttr("SingleTaskGP");
        using var gpKwargs = new PyDict();
        gpKwargs["outcome_transform"] = standardize;
        PyObject model = singleTaskGpCtor.Invoke(new[] { trainXNormalized, trainY }, gpKwargs);

        PyObject mlls = PythonRuntime.Import("gpytorch.mlls");
        using PyObject emllCtor = mlls.GetAttr("ExactMarginalLogLikelihood");
        using PyObject likelihood = model.GetAttr("likelihood");
        using PyObject mll = emllCtor.Invoke(likelihood, model);

        PyObject fit = PythonRuntime.Import("botorch.fit");
        using PyObject fitGpytorchMll = fit.GetAttr("fit_gpytorch_mll");
        fitGpytorchMll.Invoke(mll).Dispose();

        return model;
    }

    /// <summary><c>SobolQMCNormalSampler(torch.Size((num_samples,)))</c> (botorch &gt;= 0.8.0).</summary>
    private static PyObject SobolQmcNormalSampler(int numSamples)
    {
        PyObject sampling = PythonRuntime.Import("botorch.sampling");
        using PyObject samplerCtor = sampling.GetAttr("SobolQMCNormalSampler");
        using PyObject sizeCtor = BoTorchInterop.Torch.GetAttr("Size");
        using var shapeList = new PyList();
        using (var n = new PyInt(numSamples))
            shapeList.Append(n);
        using PyObject sampleShape = sizeCtor.Invoke(shapeList);
        return samplerCtor.Invoke(sampleShape);
    }

    /// <summary>
    ///     <c>optimize_acqf(acqf, standard_bounds, q=1, num_restarts, raw_samples, options={...},
    ///     sequential=True)</c> then <c>unnormalize(candidates.detach(), bounds=bounds)</c>.
    /// </summary>
    private static PyObject OptimizeAndUnnormalize(
        PyObject acqf, PyObject bounds, int numRestarts, int rawSamples, bool nonnegative)
    {
        using PyObject standardBounds = StandardBounds(bounds);

        PyObject optim = PythonRuntime.Import("botorch.optim");
        using PyObject optimizeAcqf = optim.GetAttr("optimize_acqf");
        using var options = new PyDict();
        options["batch_limit"] = new PyInt(5);
        options["maxiter"] = new PyInt(200);
        if (nonnegative)
            options["nonnegative"] = true.ToPython();
        using var kwargs = new PyDict();
        kwargs["acq_function"] = acqf;
        kwargs["bounds"] = standardBounds;
        kwargs["q"] = new PyInt(1);
        kwargs["num_restarts"] = new PyInt(numRestarts);
        kwargs["raw_samples"] = new PyInt(rawSamples);
        kwargs["options"] = options;
        kwargs["sequential"] = true.ToPython();
        using PyObject result = optimizeAcqf.Invoke(Array.Empty<PyObject>(), kwargs);

        // optimize_acqf returns (candidates, acq_values); take the candidates.
        using PyObject candidates = result[new PyInt(0)];
        using PyObject detached = candidates.InvokeMethod("detach");

        PyObject transforms = PythonRuntime.Import("botorch.utils.transforms");
        using PyObject unnormalize = transforms.GetAttr("unnormalize");
        using var unKwargs = new PyDict();
        unKwargs["bounds"] = bounds;
        return unnormalize.Invoke(new[] { detached }, unKwargs);
    }

    /// <summary><c>standard_bounds = torch.zeros_like(bounds); standard_bounds[1] = 1</c>.</summary>
    private static PyObject StandardBounds(PyObject bounds)
    {
        using PyObject zerosLike = BoTorchInterop.Torch.GetAttr("zeros_like");
        PyObject standardBounds = zerosLike.Invoke(bounds);
        using (var one = new PyInt(1))
        using (PyObject upperValue = 1.ToPython())
            standardBounds.SetItem(one, upperValue);
        return standardBounds;
    }

    private static long Size(PyObject tensor, int dim)
    {
        using PyObject d = dim.ToPython();
        using PyObject size = tensor.InvokeMethod("size", d);
        return size.As<long>();
    }

    private static void RequireSingleObjective(PyObject trainObj, string name)
    {
        if (Size(trainObj, -1) != 1)
            throw new NumSharp.ValueError($"Objective may only contain single values with {name}.");
    }

    private static NotSupportedException ConstraintsNotYetSupported(string func)
        => new($"Constrained optimization in {func} is not ported yet (Phase C). " +
               "Use an unconstrained single-objective study for now.");
}

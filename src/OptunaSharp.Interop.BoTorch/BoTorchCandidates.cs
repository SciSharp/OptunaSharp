using System;
using Python.Runtime;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     A function that suggests the next candidate — the OptunaSharp analog of BoTorch's
///     <c>candidates_func</c>. It takes the training data, objectives, constraints, the search-space
///     bounds and the pending (running) points as <c>torch.Tensor</c>s and returns the next candidates
///     as a <c>torch.Tensor</c>. When no constraint is set, <paramref name="trainCon" /> is
///     <c>null</c>; when there are no pending trials, <paramref name="pendingX" /> is <c>null</c>. The
///     function assumes <b>maximization</b> of the objective.
///     <para>Invoked under the GIL (inside a <see cref="PythonNet.PythonRuntime.Run(Action)" /> body,
///     within BoTorch's <c>manual_seed</c> scope). See the built-ins on
///     <see cref="BoTorchCandidates" /> for reference implementations, and BoTorch's homepage for how to
///     write your own.</para>
/// </summary>
public delegate PyObject BoTorchCandidatesFunc(
    PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX);

/// <summary>
///     The built-in <c>candidates_func</c>s — faithful C# ports of the functions in
///     <c>optuna_integration/botorch/botorch.py</c>, each orchestrating the <b>real BoTorch</b>
///     (<c>SingleTaskGP</c> → <c>fit_gpytorch_mll</c> → an acquisition function → <c>optimize_acqf</c>)
///     over <c>torch.Tensor</c>s via pythonnet. Every method must be called under the GIL.
///     <para><b>Phase A</b> ships the single-objective path (log-EI default + the q-batch variants).
///     Multi-objective, constrained, and knowledge-gradient variants follow in later phases; the
///     constrained branch of each function currently raises <see cref="NotSupportedException" />.</para>
///     <para><b>Cached-module discipline:</b> <see cref="BoTorchInterop.Import" /> / <c>Torch</c> return
///     per-process cached modules — used as plain locals, never disposed; only fresh
///     <c>GetAttr</c>/<c>Invoke</c> results are wrapped in <c>using</c>.</para>
/// </summary>
public static class BoTorchCandidates
{
    /// <summary>
    ///     Log Expected Improvement (LogEI) — the default <c>candidates_func</c> for single-objective
    ///     optimization. Port of <c>logei_candidates_func</c>.
    /// </summary>
    public static PyObject LogEi(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "logEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("logei_candidates_func");

        PyObject trainY = trainObj;
        using PyObject bestF = trainObj.InvokeMethod("max");
        using PyObject normX = Normalize(trainX, bounds);
        using PyObject model = FitSingleTaskGp(normX, trainY);

        PyObject analytic = BoTorchInterop.Import("botorch.acquisition.analytic");
        using PyObject logEiCtor = analytic.GetAttr("LogExpectedImprovement");
        using var acqfKwargs = new PyDict();
        acqfKwargs["model"] = model;
        acqfKwargs["best_f"] = bestF;
        using PyObject acqf = logEiCtor.Invoke(Array.Empty<PyObject>(), acqfKwargs);

        return OptimizeAndUnnormalize(acqf, bounds, numRestarts: 10, rawSamples: 512, nonnegative: false);
    }

    /// <summary>
    ///     Quasi-MC batch Expected Improvement (qEI) — the default when running trials are considered.
    ///     Port of <c>qei_candidates_func</c>.
    /// </summary>
    public static PyObject Qei(
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

        PyObject mc = BoTorchInterop.Import("botorch.acquisition.monte_carlo");
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

    /// <summary>
    ///     Quasi-MC batch Log Expected Improvement (qLogEI) — the numerically stable drop-in for
    ///     <see cref="Qei" />. Port of <c>qlogei_candidates_func</c>.
    /// </summary>
    public static PyObject QLogEi(
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

        PyObject logei = BoTorchInterop.Import("botorch.acquisition.logei");
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

    /// <summary>
    ///     Quasi-MC batch Noisy Expected Improvement (qNEI) — may beat qEI on noisy objectives. Port of
    ///     <c>qnei_candidates_func</c>.
    /// </summary>
    public static PyObject QNei(
        PyObject trainX, PyObject trainObj, PyObject? trainCon, PyObject bounds, PyObject? pendingX)
    {
        RequireSingleObjective(trainObj, "qNEI");
        if (trainCon is not null)
            throw ConstraintsNotYetSupported("qnei_candidates_func");

        PyObject trainY = trainObj;
        using PyObject normX = Normalize(trainX, bounds);
        using PyObject? normPending = MaybeNormalize(pendingX, bounds);
        using PyObject model = FitSingleTaskGp(normX, trainY);

        PyObject mc = BoTorchInterop.Import("botorch.acquisition.monte_carlo");
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

    /// <summary>
    ///     Choose the default <c>candidates_func</c> from the number of objectives and whether
    ///     constraints/running trials are present — port of <c>_get_default_candidates_func</c>.
    ///     <para>Phase A implements the single-objective branches (running → <see cref="Qei" />, else
    ///     <see cref="LogEi" />); the multi-objective branches raise <see cref="NotSupportedException" />
    ///     until Phase B.</para>
    /// </summary>
    public static BoTorchCandidatesFunc GetDefaultCandidatesFunc(
        int nObjectives, bool hasConstraint, bool considerRunningTrials)
    {
        // The full Python selector (Phase B/C completes the multi-objective branches):
        //   n>3 && !con && !running -> ehvi;  n>3 -> qparego;  n>1 -> qehvi;  running -> qei;  else logei
        if (nObjectives > 1)
            throw new NotSupportedException(
                "Multi-objective BoTorchSampler (qEHVI/EHVI/qParEGO/qNEHVI) is not ported yet (Phase B). " +
                "Use a single-objective study, or pass an explicit candidatesFunc.");
        return considerRunningTrials ? Qei : LogEi;
    }

    // ---- shared botorch orchestration -----------------------------------------------------------

    /// <summary><c>normalize(x, bounds=bounds)</c> from <c>botorch.utils.transforms</c>.</summary>
    private static PyObject Normalize(PyObject x, PyObject bounds)
    {
        PyObject transforms = BoTorchInterop.Import("botorch.utils.transforms");
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

        PyObject outcome = BoTorchInterop.Import("botorch.models.transforms.outcome");
        using PyObject standardizeCtor = outcome.GetAttr("Standardize");
        using var stdKwargs = new PyDict();
        stdKwargs["m"] = new PyInt(m);
        using PyObject standardize = standardizeCtor.Invoke(Array.Empty<PyObject>(), stdKwargs);

        PyObject models = BoTorchInterop.Import("botorch.models");
        using PyObject singleTaskGpCtor = models.GetAttr("SingleTaskGP");
        using var gpKwargs = new PyDict();
        gpKwargs["outcome_transform"] = standardize;
        PyObject model = singleTaskGpCtor.Invoke(new[] { trainXNormalized, trainY }, gpKwargs);

        PyObject mlls = BoTorchInterop.Import("gpytorch.mlls");
        using PyObject emllCtor = mlls.GetAttr("ExactMarginalLogLikelihood");
        using PyObject likelihood = model.GetAttr("likelihood");
        using PyObject mll = emllCtor.Invoke(likelihood, model);

        PyObject fit = BoTorchInterop.Import("botorch.fit");
        using PyObject fitGpytorchMll = fit.GetAttr("fit_gpytorch_mll");
        fitGpytorchMll.Invoke(mll).Dispose();

        return model;
    }

    /// <summary><c>SobolQMCNormalSampler(torch.Size((num_samples,)))</c> (botorch &gt;= 0.8.0).</summary>
    private static PyObject SobolQmcNormalSampler(int numSamples)
    {
        PyObject sampling = BoTorchInterop.Import("botorch.sampling");
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

        PyObject optim = BoTorchInterop.Import("botorch.optim");
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

        PyObject transforms = BoTorchInterop.Import("botorch.utils.transforms");
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using NumSharp;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     A sampler that uses <b>BoTorch</b>, the Bayesian-optimization library built on PyTorch — a
///     native OptunaSharp <see cref="BaseSampler" /> that drives the <b>real</b> BoTorch in-process via
///     pythonnet. Faithful port of <c>optuna_integration.botorch.BoTorchSampler</c>: the sampler
///     protocol (search-space intersection, one-hot transform, startup trials, running-trial handling,
///     direction flipping, seeding) is OptunaSharp C#, while the numerics (the GP fit and acquisition
///     optimization inside the selected <c>candidates_func</c>) are genuine BoTorch calls. Parameters
///     are transformed to continuous space and passed to BoTorch, then transformed back; categorical
///     parameters are one-hot encoded.
///
///     <para>Unlike Python, this does <b>not</b> require the <c>optuna_integration</c> package (we port
///     it); install only <c>botorch</c> (which pulls torch + gpytorch) into a discoverable CPython.</para>
///
///     <para><b>Scope:</b> the full <c>optuna_integration.BoTorchSampler</c> surface — single-objective
///     (LogEI / qEI / qLogEI / qNEI), multi-objective (qEHVI / EHVI / qNEHVI / qParEGO / qLogEI-ParEGO),
///     constrained optimization (via <c>Trial.SetConstraint</c> or the deprecated
///     <c>constraintsFunc</c>), knowledge-gradient acquisitions (qKG / qHVKG), and a CPU/GPU
///     <c>device</c> — all selected through the <see cref="BoTorchCandidatesFunc" /> enum (see
///     <see cref="BoTorchCandidates" />).</para>
///
///     <para><b>Reproducibility.</b> The first <see cref="_nStartupTrials" /> trials use an independent
///     <see cref="RandomSampler" /> (bit-exact with Optuna); from then on each acquisition optimization
///     runs inside BoTorch's <c>manual_seed(seed)</c>, with the same per-trial seed progression as
///     Optuna. Numerical agreement also depends on matching the Python, BoTorch, PyTorch, and device
///     environment; a seed alone does not make different numerical runtimes interchangeable.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new BoTorchSampler(seed: 0));
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10) - 2, 2), nTrials: 30);
///     </code>
/// </example>
/// <seealso href="https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.BoTorchSampler.html">Official Optuna integration API contract.</seealso>
public sealed class BoTorchSampler : BaseSampler
{
    private readonly bool _considerRunningTrials;
    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private readonly IntersectionSearchSpace _searchSpace = new();
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private readonly string? _device;

    // The acquisition selection from the ctor, and the resolved implementation cached on first use
    // (Python caches self._candidates_func the same way — the explicit one, else the auto-picked default).
    private readonly BoTorchCandidatesFunc _selection;
    private NativeCandidatesFunc? _candidatesFunc;

    private int? _seed;
    private int? _studyId;

    /// <param name="candidatesFunc">
    ///     The BoTorch acquisition function to optimize. Defaults to <see cref="BoTorchCandidatesFunc.Auto" />,
    ///     which chooses automatically from the number of objectives and whether running trials are
    ///     considered (single-objective: log-Expected Improvement, or qEI when running trials are
    ///     considered). All built-ins assume <b>maximization</b>.
    /// </param>
    /// <param name="nStartupTrials">
    ///     Number of initial trials sampled independently before BoTorch takes over. Defaults to 10.
    /// </param>
    /// <param name="considerRunningTrials">
    ///     If <c>true</c>, the acquisition function considers the running trials (those already fully
    ///     suggested but not yet evaluated), which can improve parallel optimization. Experimental
    ///     (added in Optuna v3.2.0).
    /// </param>
    /// <param name="constraintsFunc">
    ///     An optional function that computes the objective constraints from a completed
    ///     <see cref="FrozenTrial" />; a value strictly larger than 0 marks a violated constraint, a value
    ///     ≤ 0 is feasible. <b>Deprecated in v5.0.0</b> (removal scheduled for v7.0.0) — prefer
    ///     <c>Trial.SetConstraint</c>. When supplied, the sampler is stateful and <b>must not be reused
    ///     across studies</b>; create a new instance per study.
    /// </param>
    /// <param name="independentSampler">
    ///     Sampler used for the initial trials and for conditional parameters. Defaults to a
    ///     <see cref="RandomSampler" /> with the same <paramref name="seed" />.
    /// </param>
    /// <param name="seed">Seed for the random number generator. Defaults to <c>null</c>.</param>
    /// <param name="device">
    ///     The torch device that stores BoTorch's input/output tensors — <c>"cpu"</c> (the default when
    ///     <c>null</c>), <c>"cuda"</c>, <c>"cuda:0"</c>, … Set a CUDA device for faster sampling. Byte-parity
    ///     against Python holds only for an identical device + torch build; the CPU path is the zero-copy
    ///     parity path.
    /// </param>
    /// <summary>Creates a sampler that fits and optimizes real BoTorch acquisitions through embedded CPython.</summary>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">CPython or the required Python package cannot be initialized or imported.</exception>
    public BoTorchSampler(
        BoTorchCandidatesFunc candidatesFunc = BoTorchCandidatesFunc.Auto,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        int nStartupTrials = 10,
        bool considerRunningTrials = false,
        BaseSampler? independentSampler = null,
        int? seed = null,
        string? device = null)
    {
        // Optuna decorates BoTorchSampler with @experimental_class("2.4.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // this sampler as stable, so that warning is intentionally omitted (deliberate divergence).

        // botorch.py's `_imports.check()` — fail fast with a helpful message if BoTorch is missing.
        PythonRuntime.Run(Python.botorch.check);

        // `constraints_func` is deprecated in v5.0.0 (removal 7.0.0) — mirror Optuna's FutureWarning.
        if (constraintsFunc != null)
            OptunaWarnings.Warn(
                string.Format(Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0")
                + " Use `optuna.trial.Trial.set_constraint` instead.",
                WarningCategory.FutureWarning);

        _selection = candidatesFunc;
        _constraintsFunc = constraintsFunc;
        _considerRunningTrials = considerRunningTrials;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _seed = seed;
        _device = device;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The stateful BoTorch sampler is reused with a different study.</exception>
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        if (_studyId is null)
            _studyId = study.StudyId;
        if (_studyId != study.StudyId)
            // InMemoryStorage always returns the same study ID, so this check is a no-op there — matching Optuna.
            throw new RuntimeError("BoTorchSampler cannot handle multiple studies.");

        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // The built-in candidates_func cannot handle single-value distributions; such params are
            // sampled in the Trial. Skip them here.
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">A completed trial's constraint names differ from the first completed trial's constraint names.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var completedTrials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        var runningTrials = study
            .GetTrials(deepcopy: false, states: new[] { TrialState.Running })
            .Where(t => t.Number != trial.Number)
            .ToList();

        int nCompleted = completedTrials.Count;
        int nTrials = nCompleted + runningTrials.Count;   // Optuna counts running trials toward the startup gate.
        if (nTrials < _nStartupTrials)
            return new OrderedDict<string, object?>();

        var trans = new SearchSpaceTransform(searchSpace);
        var directions = study.Directions;
        int nObjectives = directions.Count;

        double[,] rawBounds = trans.RawBounds;   // (n_bounds, 2); transform01 is false (the default).
        int nBounds = rawBounds.GetLength(0);

        // Completed trials → maximization-signed objectives + transformed params (+ the constraint matrix).
        var completedParams = new double[nCompleted, nBounds];
        var completedValues = new double[nCompleted, nObjectives];
        List<string>? constraintKeys = null;
        double[,]? con = null;
        for (int i = 0; i < nCompleted; i++)
        {
            FrozenTrial t = completedTrials[i];
            double[] row = trans.Transform(t.Params).ToArray<double>();
            for (int d = 0; d < nBounds; d++)
                completedParams[i, d] = row[d];

            IReadOnlyList<double> tv = t.Values!;
            for (int j = 0; j < nObjectives; j++)
            {
                double value = tv[j];
                if (directions[j] == StudyDirection.Minimize)   // BoTorch always assumes maximization.
                    value *= -1;
                completedValues[i, j] = value;
            }

            // Constraint matrix (botorch.py:1240-1256): the constraint names are fixed by the FIRST
            // completed trial; every later trial must carry the identical set, else it is a hard error.
            // con is (n_completed, n_constraints), NaN-filled up front like numpy.full.
            OrderedDict<string, double> constraints = t.Constraints;
            if (constraintKeys is null)
            {
                constraintKeys = constraints.Keys.ToList();
                if (constraintKeys.Count != 0)
                {
                    con = new double[nCompleted, constraintKeys.Count];
                    for (int r = 0; r < nCompleted; r++)
                        for (int c = 0; c < constraintKeys.Count; c++)
                            con[r, c] = double.NaN;
                }
            }
            else if (!new HashSet<string>(constraints.Keys).SetEquals(constraintKeys))
            {
                throw new RuntimeError(
                    $"Expected the constraints named {PyListRepr(constraintKeys)} " +
                    $"but received {PyListRepr(constraints.Keys)}.");
            }
            if (con is not null)
                for (int c = 0; c < constraintKeys!.Count; c++)
                    con[i, c] = constraints[constraintKeys[c]];
        }

        // botorch.py:1265 — constraints_func was supplied but nothing correctly computed a constraint.
        if (_constraintsFunc is not null && con is null)
            OptunaWarnings.Warn(
                "`constraints_func` was given but no call to it correctly computed constraints. " +
                "Constraints passed to `candidates_func` will be `None`.");

        // Running trials (only when considered): the ones whose full param set is already suggested.
        // (Python fills the rest with NaN and drops them; skipping non-present rows is the same set.)
        double[,]? runningParams = null;
        if (_considerRunningTrials)
        {
            var rows = new List<double[]>();
            foreach (FrozenTrial t in runningTrials)
                if (searchSpace.Keys.All(k => t.Params.ContainsKey(k)))
                    rows.Add(trans.Transform(t.Params).ToArray<double>());
            if (rows.Count > 0)
            {
                runningParams = new double[rows.Count, nBounds];
                for (int i = 0; i < rows.Count; i++)
                    for (int d = 0; d < nBounds; d++)
                        runningParams[i, d] = rows[i][d];
            }
        }

        // bounds tensor is transposed to (2, n_bounds) — botorch.py's `bounds.transpose_(0, 1)`.
        var boundsTransposed = new double[2, nBounds];
        for (int d = 0; d < nBounds; d++)
        {
            boundsTransposed[0, d] = rawBounds[d, 0];
            boundsTransposed[1, d] = rawBounds[d, 1];
        }

        // Resolve the candidates_func once (Python caches self._candidates_func the same way): the
        // explicit selection, or the Auto default from n_objectives / constraint / running-trials.
        _candidatesFunc ??= BoTorchCandidates.Resolve(
            _selection, nObjectives, hasConstraint: con is not null, _considerRunningTrials);

        NDArray candidate = PythonRuntime.Run(() =>
        {
            using PyObject trainX = BoTorchInterop.ToTensor(completedParams, _device);
            using PyObject trainObj = BoTorchInterop.ToTensor(completedValues, _device);
            using PyObject boundsTensor = BoTorchInterop.ToTensor(boundsTransposed, _device);
            PyObject? conTensor = con is null ? null : BoTorchInterop.ToTensor(con, _device);
            PyObject? pendingTensor = runningParams is null ? null : BoTorchInterop.ToTensor(runningParams, _device);
            try
            {
                // Each acquisition optimization runs inside BoTorch's manual_seed(seed) scope.
                PyObject cand;
                using (Python.botorch.manual_seed(_seed))
                    cand = _candidatesFunc!(trainX, trainObj, conTensor, boundsTensor, pendingTensor);
                // Advance the seed per trial (Optuna does `self._seed += 1` inside the manual_seed scope).
                if (_seed is { } seedValue)
                    _seed = seedValue + 1;
                try
                {
                    return ExtractCandidate(cand, nBounds);
                }
                finally
                {
                    cand.Dispose();
                }
            }
            finally
            {
                conTensor?.Dispose();
                pendingTensor?.Dispose();
            }
        });

        // Un-transform back to the actual parameter values (in the search-space key order).
        return trans.Untransform(candidate);
    }

    /// <summary>
    ///     Validate the candidates tensor (botorch.py's shape checks + <c>squeeze(0)</c>) and cross it to a
    ///     1-D NumSharp <see cref="NDArray" /> through NumSharp's zero-copy interop. Under the GIL.
    /// </summary>
    /// <param name="candidates">The borrowed result expected to be a torch tensor containing exactly one candidate.</param>
    /// <param name="nBounds">The number of transformed parameter coordinates required by the search-space bounds.</param>
    /// <returns>An independent one-dimensional array containing the candidate's transformed coordinates.</returns>
    /// <exception cref="NumSharp.TypeError">The acquisition result is not a torch.Tensor.</exception>
    /// <exception cref="NumSharp.ValueError">The tensor rank, batch size, or coordinate count does not describe one candidate with the expected bounds.</exception>
    private static NDArray ExtractCandidate(PyObject candidates, int nBounds)
    {
        if (!candidates.HasAttr("dim"))
            throw new TypeError("Candidates must be a torch.Tensor.");

        long dim = Python.torch.Tensor.dim_t(candidates);
        PyObject work = candidates;
        bool disposeWork = false;
        try
        {
            if (dim == 2)
            {
                if (Python.torch.Tensor.size_t(candidates, 0) != 1)
                {
                    using PyObject shape = Python.torch.Tensor.size(candidates);
                    throw new ValueError(
                        "Candidates batch optimization is not supported and the first dimension must " +
                        $"have size 1 if candidates is a two-dimensional tensor. Actual: {shape}.");
                }
                work = Python.torch.Tensor.squeeze(candidates, 0);
                disposeWork = true;
            }

            if (Python.torch.Tensor.dim_t(work) != 1)
                throw new ValueError("Candidates must be one or two-dimensional.");
            long candSize = Python.torch.Tensor.size_t(work, 0);
            if (candSize != nBounds)
                throw new ValueError(
                    "Candidates size must match with the given bounds. Actual candidates: " +
                    $"{candSize}, bounds: {nBounds}.");

            return BoTorchInterop.ToNDArray(work);
        }
        finally
        {
            if (disposeWork)
                work.Dispose();
        }
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);

    /// <inheritdoc />
    public override void ReseedRng()
    {
        _independentSampler.ReseedRng();
        if (_seed is not null)
        {
            // Python: self._seed = numpy.random.RandomState().randint(numpy.iinfo(numpy.int32).max).
            // The reseeded value is OS-entropy random; draw it from the crypto RNG (as GPSampler does).
            Span<byte> buffer = stackalloc byte[4];
            RandomNumberGenerator.Fill(buffer);
            _seed = (int)(BitConverter.ToUInt32(buffer) % (uint)int.MaxValue);
        }
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        // botorch.py's after_trial: compute + store the deprecated constraints_func's values, then delegate.
        if (_constraintsFunc != null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
        _independentSampler.AfterTrial(study, trial, state, values);
    }

    /// <summary>Format a key set as Python's <c>list</c> repr (<c>['0', '1']</c>) for the verbatim
    /// constraint-mismatch <see cref="RuntimeError" /> message.</summary>
    /// <param name="keys">The ordered constraint names included in the diagnostic.</param>
    /// <returns>The bracketed, quoted list of constraint names.</returns>
    private static string PyListRepr(IEnumerable<string> keys)
        => "[" + string.Join(", ", keys.Select(k => "'" + k + "'")) + "]";
}

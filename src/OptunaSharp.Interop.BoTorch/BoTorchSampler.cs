using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using NumSharp;
using Python;
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
///     <para><b>Scope (Phase A):</b> single-objective optimization (default: log-Expected Improvement;
///     the q-batch variants when running trials are considered). Multi-objective, constrained, and
///     knowledge-gradient variants land in later phases (see <see cref="BoTorchCandidates" />).</para>
///
///     <para><b>Reproducibility.</b> The first <see cref="_nStartupTrials" /> trials use an independent
///     <see cref="RandomSampler" /> (bit-exact with Optuna); from then on each acquisition optimization
///     runs inside BoTorch's <c>manual_seed(seed)</c>, and the seed is advanced per trial exactly as
///     Optuna does — so an OptunaSharp study reproduces <c>optuna_integration.BoTorchSampler</c>
///     trial-for-trial on the same seed.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new BoTorchSampler(seed: 0));
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10) - 2, 2), nTrials: 30);
///     </code>
/// </example>
public sealed class BoTorchSampler : BaseSampler
{
    private readonly bool _considerRunningTrials;
    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private readonly IntersectionSearchSpace _searchSpace = new();

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
    /// <param name="independentSampler">
    ///     Sampler used for the initial trials and for conditional parameters. Defaults to a
    ///     <see cref="RandomSampler" /> with the same <paramref name="seed" />.
    /// </param>
    /// <param name="seed">Seed for the random number generator. Defaults to <c>null</c>.</param>
    public BoTorchSampler(
        BoTorchCandidatesFunc candidatesFunc = BoTorchCandidatesFunc.Auto,
        int nStartupTrials = 10,
        bool considerRunningTrials = false,
        BaseSampler? independentSampler = null,
        int? seed = null)
    {
        // @experimental_class("2.4.0") — the whole sampler is experimental.
        OptunaWarnings.Warn(
            "BoTorchSampler is experimental (supported from v2.4.0). The interface can change in the future.",
            WarningCategory.ExperimentalWarning);

        // botorch.py's `_imports.check()` — fail fast with a helpful message if BoTorch is missing.
        PythonRuntime.Run(botorch.check);

        _selection = candidatesFunc;
        _considerRunningTrials = considerRunningTrials;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _seed = seed;
    }

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

        // Completed trials → maximization-signed objectives + transformed params.
        var completedParams = new double[nCompleted, nBounds];
        var completedValues = new double[nCompleted, nObjectives];
        for (int i = 0; i < nCompleted; i++)
        {
            FrozenTrial t = completedTrials[i];
            double[] row = trans.Transform(t.ParamsDict).ToArray<double>();
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
        }

        // Running trials (only when considered): the ones whose full param set is already suggested.
        // (Python fills the rest with NaN and drops them; skipping non-present rows is the same set.)
        double[,]? runningParams = null;
        if (_considerRunningTrials)
        {
            var rows = new List<double[]>();
            foreach (FrozenTrial t in runningTrials)
                if (searchSpace.Keys.All(k => t.Params.ContainsKey(k)))
                    rows.Add(trans.Transform(t.ParamsDict).ToArray<double>());
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

        // Phase A: no constraints (train_con is always null).
        _candidatesFunc ??= BoTorchCandidates.Resolve(
            _selection, nObjectives, hasConstraint: false, _considerRunningTrials);

        NDArray candidate = PythonRuntime.Run(() =>
        {
            using PyObject trainX = BoTorchInterop.ToTensor(completedParams);
            using PyObject trainObj = BoTorchInterop.ToTensor(completedValues);
            using PyObject boundsTensor = BoTorchInterop.ToTensor(boundsTransposed);
            PyObject? pendingTensor = runningParams is null ? null : BoTorchInterop.ToTensor(runningParams);
            try
            {
                // Each acquisition optimization runs inside BoTorch's manual_seed(seed) scope.
                PyObject cand;
                using (botorch.manual_seed(_seed))
                    cand = _candidatesFunc!(trainX, trainObj, null, boundsTensor, pendingTensor);
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
    private static NDArray ExtractCandidate(PyObject candidates, int nBounds)
    {
        if (!candidates.HasAttr("dim"))
            throw new TypeError("Candidates must be a torch.Tensor.");

        long dim = candidates.dim();
        PyObject work = candidates;
        bool disposeWork = false;
        try
        {
            if (dim == 2)
            {
                if (candidates.size(0) != 1)
                {
                    using PyObject shape = candidates.size();
                    throw new ValueError(
                        "Candidates batch optimization is not supported and the first dimension must " +
                        $"have size 1 if candidates is a two-dimensional tensor. Actual: {shape}.");
                }
                work = candidates.squeeze(0);
                disposeWork = true;
            }

            if (work.dim() != 1)
                throw new ValueError("Candidates must be one or two-dimensional.");
            long candSize = work.size(0);
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

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);

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

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _independentSampler.AfterTrial(study, trial, state, values);
}

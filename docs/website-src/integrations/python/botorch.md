# BoTorch

`OptunaSharp.Interop.BoTorch` makes the actual BoTorch/PyTorch/GPyTorch stack available as an
OptunaSharp `BaseSampler`. C# handles trial history, the search-space transform and sampler protocol;
Python fits the Gaussian-process model and optimizes the selected acquisition function.

## Install and optimize

```shell
dotnet add package OptunaSharp.Interop.BoTorch
python -m pip install botorch
```

The package targets .NET 8 and .NET 10 and brings the shared [Python.NET runtime](pythonnet.md)
transitively. Install BoTorch and its dependencies into that runtime's interpreter. The Python
`optuna_integration` package is not required by this bridge; its glue is implemented in C#.

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Interop.BoTorch;

var sampler = new BoTorchSampler(seed: 0, nStartupTrials: 10);
var study = Optuna.CreateStudy(sampler: sampler);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 30, nJobs: 1);

Console.WriteLine(study.BestValue);
```

Construction imports BoTorch immediately, so missing dependencies fail before optimization starts.
The sampler emits an experimental warning, matching the ported integration.

## Acquisition functions

Select an acquisition with the public `BoTorchCandidatesFunc` enum:

| Value | Purpose |
|---|---|
| `Auto` | Select from objective count, constraints and pending-trial handling |
| `LogEi` | Single-objective log expected improvement |
| `Qei` | Monte Carlo expected improvement, including pending points |
| `QLogEi` | Log expected improvement using Monte Carlo sampling |
| `QNei` | Noisy expected improvement |
| `Qehvi` | Expected hypervolume improvement |
| `Ehvi` | Analytic expected hypervolume improvement; rejects constraints |
| `Qnehvi` | Noisy expected hypervolume improvement |
| `QParEgo` | Random scalarization with expected improvement |
| `QLogEiParEgo` | Random scalarization with log expected improvement |
| `Qkg` | Single-objective knowledge gradient |
| `Qhvkg` | Hypervolume knowledge gradient |

The exact `Auto` dispatch is:

1. More than three objectives, no constraints and no running-trial consideration: `Ehvi`.
2. More than three objectives otherwise: `QParEgo`.
3. Two or three objectives: `Qehvi`.
4. One objective with `considerRunningTrials: true`: `Qei`.
5. One objective otherwise: `LogEi`.

The resolved function is cached on first use. `Qkg` and `Qhvkg` are explicit choices only. Acquisition
availability also depends on the installed BoTorch version; implementations include feature/version
guards. The public API accepts the enum, not a custom delegate or raw `torch.Tensor` callback.

Despite the `q` acquisition names, this sampler returns one candidate per OptunaSharp trial. It does
not expose a multi-candidate batch-return API.

## Multiple objectives and constraints

```csharp
using OptunaSharp;
using OptunaSharp.Interop.BoTorch;

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new BoTorchSampler(candidatesFunc: BoTorchCandidatesFunc.Qehvi, seed: 0));

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", 0, 1);
    double y = trial.SuggestFloat("y", 0, 1);
    trial.SetConstraint("budget", x + y - 1.2);
    return new[] { x * x + y * y, (1 - x) * (1 - x) + y * y };
}, nTrials: 30, nJobs: 1);
```

A constraint is feasible at a value less than or equal to zero. Record the same constraint names on
every completed trial, including startup trials: the first completed trial defines the columns, and
a later different set raises `RuntimeError`. The legacy
`constraintsFunc: Func<FrozenTrial, IEnumerable<double>>` constructor argument remains implemented,
but emits the upstream 5.0.0 deprecation warning; prefer `Trial.SetConstraint`.

BoTorch internally maximizes every objective. The bridge negates values for each minimizing direction
before building training tensors and leaves values stored in the C# study in their original meaning.

## Startup, search space and pending trials

`nStartupTrials` defaults to 10. The startup gate counts completed **and running** trials, even when
`considerRunningTrials` is false. Startup and parameters outside the intersection search space use
`independentSampler`, which defaults to a `RandomSampler` seeded with the same seed. Single-value
distributions are excluded from the relative space.

Continuous, integer, logarithmic, stepped and categorical distributions pass through OptunaSharp's
search-space transform; categorical parameters use one-hot columns. Conditional parameters absent
from the intersection are sampled independently. Completed trials provide training observations.

With `considerRunningTrials: true`, other running trials whose entire relative parameter set has
already been suggested are sent as pending points. Partially specified trials are omitted. This
supports pending-aware acquisitions; it does not make concurrent suggestions deterministic or
serialize the C# sampler's mutable seed and cache. Use one sampler per study, and sequential execution
when matching a reference trajectory.

## Devices, memory and lifetime

`device: null` or `"cpu"` selects CPU tensors. `device: "cuda"` or `"cuda:0"` requests a PyTorch
device, requiring a suitable PyTorch build and available hardware. CPU input matrices are float64
views over pinned managed buffers through NumSharp, NumPy and torch. A non-CPU device requires a copy
to that device; the result is copied back into an independent managed `NDArray` before Python handles
are disposed.

The sampler handles the GIL and tensor lifetime internally and does not implement `IDisposable`.
Each acquisition optimization runs within BoTorch's `manual_seed` scope, then advances the seed by
one. Its search-space state, seed and resolved acquisition belong to one study; create a new instance
for another study. As with the other bridges, embedded CPython is a process-lifetime dependency.

## Validation and limits

The [sampler parity tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/BoTorchSamplerParityTests.cs)
compare sampled parameters and study state against live `optuna_integration.BoTorchSampler`, including
continuous, integer and categorical parameters, several acquisition functions, multiple objectives
and constraints. [Integration tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/BoTorchIntegrationParityTests.cs)
cover pending points, the legacy constraint function and mixed parameter types.
[Acquisition tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/BoTorchApiParityTests.cs)
cover the explicit acquisition suite, selection rules, float64 crossing and error cases.

Byte comparisons are against the same installed Python stack and seeded CPU computation. CUDA has
a conditional smoke test for valid candidates, not a CPU-versus-GPU byte-parity promise. Python tests
report inconclusive when their dependencies are unavailable; missing CUDA likewise skips the device
smoke test. Reproducibility across different BoTorch, torch, device or numerical-library versions is
not established by these tests.

Source: [sampler](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.BoTorch/BoTorchSampler.cs),
[acquisition implementations](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.BoTorch/BoTorchCandidates.cs),
[tensor crossing](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.BoTorch/BoTorchInterop.cs).

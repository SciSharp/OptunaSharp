# OptunaSharp.Interop.BoTorch

`BoTorchSampler` for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a native OptunaSharp
`BaseSampler` that drives the **real [BoTorch](https://botorch.org/)** (PyTorch + GPyTorch) library
in-process via [Python.NET](https://github.com/pythonnet/pythonnet).

It is a faithful C# port of the *glue* in `optuna_integration.BoTorchSampler` — the sampler protocol
(search-space intersection, one-hot transform, startup trials, running-trial handling, direction
flipping, seed handling) and the `candidates_func` selection — while every **numerical** step
(`SingleTaskGP`, `fit_gpytorch_mll`, the acquisition functions, `optimize_acqf`) is a **direct call
into `botorch`**. So you get the genuine article: **byte-exact with BoTorch**, the full acquisition
suite, a custom `candidates_func`, and `device=` GPU offload — with the Optuna side staying native
OptunaSharp (your study, storage, and RNG).

Unlike the Python integration, this package does **not** depend on `optuna_integration` — we port it.
You only need `botorch` installed in a discoverable CPython:

```bash
pip install botorch          # pulls torch + gpytorch
```

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Interop.BoTorch;

var study = Optuna.CreateStudy(sampler: new BoTorchSampler(seed: 0));
study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10) - 2, 2), nTrials: 30);

Console.WriteLine($"{study.BestValue} at {study.BestParams["x"]}");
```

By default the acquisition function is chosen automatically from the number of objectives and whether
constraints are present (single-objective → log-Expected Improvement). Pick one explicitly, or supply
your own:

```csharp
var study = Optuna.CreateStudy(
    sampler: new BoTorchSampler(candidatesFunc: BoTorchCandidates.Qnei, seed: 0));
```

## How it works

`BoTorchSampler` runs the first `nStartupTrials` (default 10) through an independent `RandomSampler`
(bit-exact with Optuna). From then on, on each ask it assembles the completed (and, if
`considerRunningTrials`, running) trials into the `train_x` / `train_obj` / `bounds` tensors exactly as
`optuna_integration.botorch` does, wraps the call in BoTorch's `manual_seed`, and hands them to the
selected `candidates_func`, which builds and fits a `SingleTaskGP` and optimizes the acquisition
function — all in real BoTorch. The returned candidate is un-transformed back to your parameters.

## Requirements

- A discoverable CPython (set `PYTHONNET_PYDLL` to pin the interpreter) with `pip install botorch`.
- Not AOT-compatible (it hosts CPython). This is a satellite package; OptunaSharp Core stays pure
  managed.

Part of the [SciSharp STACK](https://github.com/SciSharp).

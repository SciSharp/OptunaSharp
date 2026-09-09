# OptunaSharp.Interop.PyCma

`PyCmaSampler` for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a native OptunaSharp
`BaseSampler` that drives the **real [pycma](https://github.com/CMA-ES/pycma)** (Nikolaus Hansen's
`cma` package: `cma.CMAEvolutionStrategy` + `cma.BoundTransform`) in-process via
[Python.NET](https://github.com/pythonnet/pythonnet).

It is a faithful C# port of the *glue* in `optuna_integration.PyCmaSampler` — "Layer 1": the sampler
protocol (intersection search space, startup trials, direction flipping), the parameter ⇄ cma-space
transforms, and the `ask`/`tell` generation replay — while the **CMA-ES numerics** (the covariance
update and candidate generation) are **direct calls into `cma`**. So a seeded run is **byte-exact with
pycma**, with the Optuna side staying native OptunaSharp (your study, storage, and RNG). The bridge keeps
pythonnet off its public surface.

Unlike the Python integration, this package does **not** depend on `optuna_integration` — we port it.
You only need `cma` installed in a discoverable CPython:

```bash
pip install cma
```

## Not the same as `CmaEsSampler`

OptunaSharp's native `CmaEsSampler` (in Core) ports the separate
[`cmaes`](https://github.com/CyberAgentAILab/cmaes) package — a **different** CMA-ES implementation.
`PyCmaSampler` wraps the `cma` (pycma) package. This mirrors Optuna's own split between
`optuna.samplers.CmaEsSampler` (cmaes) and `optuna.integration.PyCmaSampler` (pycma). The two are **not
bit-identical to each other**; each is byte-exact against its own backend. `PyCmaSampler` is
**deprecated upstream** (`@deprecated_class("4.9.0", "6.0.0")`) — prefer the native `CmaEsSampler` for
new work; this package exists for faithful parity with `optuna_integration.PyCmaSampler`.

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Interop.PyCma;

var study = Optuna.CreateStudy(sampler: new PyCmaSampler(seed: 42));
study.Optimize(t =>
{
    double x = t.SuggestFloat("x", -5, 5);
    double y = t.SuggestFloat("y", -5, 5);
    return x * x + y * y;
}, nTrials: 50);

Console.WriteLine($"{study.BestValue} at ({study.BestParams["x"]}, {study.BestParams["y"]})");
```

`PyCmaSampler` runs the first `nStartupTrials` (default 1) through an independent `RandomSampler`
(bit-exact with Optuna). From then on, on each ask it rebuilds a `cma.CMAEvolutionStrategy` with the same
seed, replays every completed generation through `tell`, and asks for the next candidate — all in real
pycma. It optimizes **float, int, and categorical** parameters (categoricals as ordinals), skipping only
single-value distributions. Note that, as in Optuna, `PyCmaSampler` does not support 1-D search spaces or
multi-objective studies.

## Requirements

- A discoverable CPython (set `PYTHONNET_PYDLL` to pin the interpreter) with `pip install cma`.
- Not AOT-compatible (it hosts CPython). This is a satellite package; OptunaSharp Core stays pure
  managed.

Part of the [SciSharp STACK](https://github.com/SciSharp).

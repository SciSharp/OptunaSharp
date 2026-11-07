# Python ecosystem

OptunaSharp can run Python optimization libraries inside the .NET process while keeping the study,
trial API, objective and storage in C#. Each bridge is an optional package; installing `OptunaSharp`
alone does not install or start Python.

| Integration | .NET package | Python dependency | What runs in Python |
|---|---|---|---|
| [Python.NET runtime](pythonnet.md) | `OptunaSharp.Interop.pythonnet` | A compatible CPython | Interpreter, imports and object conversion |
| [OptunaHub](optunahub.md) | `OptunaSharp.Interop.pythonnet` | `optuna`, `optunahub`, selected package requirements | Registry samplers, pruners and benchmark problems |
| [BoTorch](botorch.md) | `OptunaSharp.Interop.BoTorch` | `botorch` and its dependencies | GP fitting and acquisition optimization |
| [pycma](pycma.md) | `OptunaSharp.Interop.PyCma` | `cma` and its dependencies | CMA-ES covariance updates and candidate generation |
| [pymoo](pymoo.md) | `OptunaSharp.Interop.Pymoo` | `pymoo` and its dependencies | Eleven evolutionary algorithms, operators and survival |

All four packages target .NET 8 and .NET 10 and share one embedded CPython runtime. BoTorch, pycma and
pymoo bring the Python.NET bridge transitively, but NuGet does not install their Python dependencies.
Start with the [runtime setup](pythonnet.md), then follow the integration's installation and example.

## Choose a bridge

Use **OptunaHub** to load a contributor package, run a benchmark with an OptunaSharp sampler, or try a
Python sampler or pruner through its supported Optuna protocol. Its shadow-study adapter has a narrower
state contract than a Python study: read the [compatibility limits](optunahub.md#compatibility-and-state)
before choosing a registry method.

Use **BoTorch** for acquisition functions implemented by the actual PyTorch/GPyTorch stack, including
multi-objective and constrained Bayesian optimization. Use **pycma** when reproducing the Python
`PyCmaSampler` integration; its numerical backend differs from the managed `CmaEsSampler`. Use
**pymoo** for the exposed multi-objective evolutionary algorithms and their reference directions,
aspiration points and genetic operators.

The [native scikit-learn backend](../training/scikit-learn.md) also reproduces part of a Python library,
but its regression and cross-validation implementation is managed C# in the core `OptunaSharp` package.
It does not use this runtime.

## What reproducibility means

A seed is one input to reproducibility. Match the Python package versions, numerical dependencies,
device and execution order as well. BoTorch and pycma have tests against their Python Optuna
integrations; pymoo has behavioral tests for all eleven exposed algorithms and direct replay comparisons
for specific configurations. Those scopes are described on each page. They do not establish identical
results across arbitrary package upgrades, CPU/GPU builds, or concurrent execution schedules.

The bridges run their Python on the runtime's single Python thread and acquire the GIL there
([the Python thread](pythonnet.md#the-python-thread)). That protects access to Python objects and makes
any calling thread and any `nJobs` safe. It does not turn a stateful sampler into a shared service: create
one sampler per study, and start with `nJobs: 1` when reproducing a reference run.

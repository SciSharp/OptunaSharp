# Introduction

**OptunaSharp** is a .NET port of [Optuna](https://github.com/optuna/optuna), the define-by-run
hyperparameter optimization framework. It aims for **1-to-1 behavioral parity with Optuna 5.x** and
is built on top of [NumSharp](https://github.com/SciSharp/NumSharp) for numerical work.

> **Status: core surface ported.** The core algorithm surface is in place — studies, trials,
> distributions, samplers, pruners, storages, importance, terminator, and visualization — with more
> landing module-by-module.

## The study / trial model

Optuna's two core concepts, which OptunaSharp mirrors:

- **Study** — an optimization session over an objective function, with a direction
  (`minimize`/`maximize`), a sampler, a pruner, and a storage.
- **Trial** — a single evaluation of the objective. During a trial the objective *suggests*
  hyperparameters (`suggest_float`, `suggest_int`, `suggest_categorical`, ...), which the sampler
  draws from the search space it has inferred so far (*define-by-run*).

```python
# Optuna (Python) — the behavior OptunaSharp targets
import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(study.best_params)   # -> {'x': ~2.0}
```

## Why a .NET port

- Use Optuna's ergonomics from C#/F# without embedding CPython in production.
- Interoperate with the rest of the [SciSharp STACK](https://github.com/SciSharp) (NumSharp,
  TensorFlow.NET, ML.NET) in one runtime.
- Deterministic, reproducible optimization runs under a fixed seed, matching Optuna.

## How parity is verified

Optuna is the **source of truth**. OptunaSharp is checked against the real library through the
`OptunaSharp.Tests.Interop` project, which runs **live Optuna in-process via pythonnet** and asserts
that OptunaSharp produces matching results — the same approach NumSharp uses against live NumPy.

Connect your trainer, storage, Python optimizer, cloud artifacts, or charts through the
[Integrations catalog](../integrations/index.md). It covers every optional package and the
training integrations built into Core.

Continue to [Getting Started](getting-started.md).

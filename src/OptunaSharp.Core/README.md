# OptunaSharp

**Optuna for .NET** — a C#/F# port of [Optuna](https://github.com/optuna/optuna), the
define-by-run hyperparameter optimization framework, built on top of
[NumSharp](https://github.com/SciSharp/NumSharp).

> **Status: early scaffolding.** This package currently ships the project stack only — the
> Optuna algorithm surface (studies, trials, samplers, pruners, storages) is not yet ported.
> It is published so the build/test/package/CI pipeline is in place and can grow incrementally.

## Install

```bash
dotnet add package OptunaSharp
```

## Compatibility target

OptunaSharp tracks **Optuna 5.x** as its behavioral source of truth: where OptunaSharp and Optuna
differ, Optuna is treated as correct and OptunaSharp is aligned to it.

## License

MIT. OptunaSharp is a derivative of Optuna (© Preferred Networks, Inc.) and retains its notice —
see `LICENSE`.

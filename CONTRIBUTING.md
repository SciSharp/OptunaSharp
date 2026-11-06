# Contributing to OptunaSharp

Thanks for your interest! OptunaSharp is a **.NET port of [Optuna](https://github.com/optuna/optuna)**.
The core surface of Optuna 5.x is ported ([`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) has the
phase-by-phase status; the `optuna.visualization.matplotlib` backend is still planned, in
[`docs/VISUALIZATION_MATPLOTLIB_PLAN.md`](docs/VISUALIZATION_MATPLOTLIB_PLAN.md)). The most useful
help now is closing parity gaps:
a behavior where OptunaSharp and Optuna disagree, reproduced as a test that asserts what Optuna does.

## The one rule: Optuna is the source of truth

OptunaSharp targets **1-to-1 behavioral parity with Optuna 5.x**. When in doubt, do what Optuna does.

1. **Run real Optuna first.** Observe the exact behavior — return values, defaults, edge cases,
   errors and their messages — by running Python. Don't guess from docs.
2. **Match behavior, then structure.** Prefer mirroring Optuna's implementation structure over
   inventing a new one, so future readers can diff the two.
3. **Edge cases matter.** Distribution bounds, NaN/inf handling, empty studies, categorical
   ordering, reproducible sampling under a fixed seed, pruning boundaries.
4. **Breaking changes are OK** when they bring us closer to Optuna.
5. **Test from Optuna output.** New tests assert against values produced by running actual Optuna —
   ideally live, through the interop harness (see below).

This mirrors how [NumSharp](https://github.com/SciSharp/NumSharp) treats NumPy as its oracle.

## Prerequisites

- .NET SDK **8.0** and **10.0** (the repo multi-targets both).
- For interop tests: a Python with the vendored Optuna installed (`git submodule update --init
  refs/optuna`, then `pip install -e refs/optuna`); optional, those tests self-skip otherwise.
- NumSharp + NumSharp.Interop.OpenBLAS come from **NuGet** (restored automatically); nothing extra to
  set up. The NumSharp source is vendored reference-only at `refs/NumSharp` — see the README,
  "The NumSharp reference".

## Build & test

```bash
dotnet build SciSharp.OptunaSharp.sln -c Release

# Unit tests (what per-push CI runs; drop `&TestCategory!=Slow` for the full run)
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython&TestCategory!=Slow"

# Live-Optuna parity (needs Python + the vendored optuna)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release
```

## Test categories

Use the typed attributes in `test/OptunaSharp.Tests/TestCategory.cs`:

| Attribute | Meaning | CI |
|-----------|---------|----|
| `[OpenBugs]` | Known-failing reproduction / not-yet-ported behavior. Remove when it passes. | excluded |
| `[Misaligned]` | An intentional, documented difference from Optuna (the test passes). | runs |
| `[RequiresPython]` | Needs live Optuna via pythonnet; self-skips when absent. | interop job only |
| `[HighMemory]` | Resource-heavy. | excluded |
| `[WindowsOnly]` | Windows-only API. | excluded off-Windows |
| `[Slow]` | ~10 s or more each. Runtime only, never to hide a failure. | excluded on push/PR; runs on `v*` tags and manual dispatch |

## Coding style

- Follow `.editorconfig` (the SciSharp house style).
- Match the naming/comment density of surrounding code.
- Every public type/member ports an Optuna concept — name it after its Optuna counterpart where it
  helps a reader cross-reference.

## Pull requests

1. Branch from `master`.
2. Keep PRs focused — ideally one Optuna module (or a slice of one) at a time.
3. Include tests derived from real Optuna behavior.
4. Ensure `dotnet build` and the unit-test filter above are green.
5. Describe **what Optuna does** and how your change matches it.

## Strong naming

Every assembly is signed with `Open.snk` (Microsoft's published open key, shared across the SciSharp
STACK) via `Directory.Build.props`. Don't remove signing or add a keyless `InternalsVisibleTo`
(it won't compile). The CI strong-naming gate asserts the packed assemblies carry the shared key.

## License

By contributing you agree your contributions are licensed under the [MIT License](LICENSE).

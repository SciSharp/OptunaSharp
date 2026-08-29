# OptunaSharp.Tests.Interop — parity against LIVE Optuna

This is the test project that runs **real CPython + Optuna embedded in the test process** (via
[pythonnet](https://github.com/pythonnet/pythonnet)). Optuna is treated as the source of truth: a
test asks the real library to compute an answer *right now* and asserts OptunaSharp matches it. It is
the direct analogue of `NumSharp.Tests.Interop`, which does the same against live NumPy.

> **Status: scaffolding.** No OptunaSharp algorithms are ported yet, so the only tests here are
> smoke tests that prove the harness works (the engine starts, Optuna imports, a real study runs).
> They are the template the real parity tests will follow.

## Moving parts

- **`PythonSession`** — one embedded engine per process (`[AssemblyInitialize]`). Discovers a Python
  (honors `PYTHONNET_PYDLL`, else probes `python`/`python3` on PATH), imports Optuna to fail fast, and
  exposes `EnsureOrInconclusive()`. A machine without Python/Optuna reports **Inconclusive** rather
  than red — unless `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE` is set (CI does this so a discovery break
  goes red, never green-by-skipping).
- **`OptunaInteropSmokeTests`** — imports Optuna, reports its `__version__`, and runs a tiny
  `create_study().optimize(...)` end-to-end.

## Running

```bash
cd test/OptunaSharp.Tests.Interop

# Needs a discoverable Python with `optuna` installed (pip install optuna)
dotnet test -f net10.0

# Point at a specific interpreter's shared library
PYTHONNET_PYDLL=/path/to/libpython3.12.so dotnet test -f net10.0

# On a box without Python: tests self-skip (Inconclusive), never fail
dotnet test -f net10.0
```

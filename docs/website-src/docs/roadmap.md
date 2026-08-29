# Porting Roadmap

OptunaSharp is being ported module-by-module, tracking Optuna's own layout. Nothing below is
implemented yet — this is the map.

## Methodology: Optuna is the source of truth

1. **Run real Optuna first** — observe exact behavior (values, defaults, edge cases, error
   messages) by running Python.
2. **Match behavior, then structure** — mirror Optuna's implementation shape where practical.
3. **Edge cases matter** — distribution bounds, categorical ordering, reproducibility under a fixed
   seed, pruning boundaries, empty studies.
4. **Test from Optuna output** — assert against values produced by the real library, ideally live
   through the interop harness.

This mirrors how [NumSharp](https://github.com/SciSharp/NumSharp) treats NumPy as its oracle.

## Modules to port

| Optuna module | Role | Status |
|---------------|------|--------|
| `distributions` | Search-space distributions (float / int / categorical) | not started |
| `trial` | `Trial` + `FrozenTrial`, the `suggest_*` API | not started |
| `study` | `Study`, `optimize`, directions, storage glue | not started |
| `samplers` | TPE, random, grid, CMA-ES, GP, QMC, ... | not started |
| `pruners` | Median, Hyperband, successive halving, threshold, ... | not started |
| `storages` | In-memory + RDB / journal persistence | not started |
| `search_space` | Search-space inference | not started |
| `importance` | Hyperparameter importance | not started |
| `terminator` | Automated termination | not started |
| `visualization` | Optimization-history plots | not started |
| `exceptions` / `logging` | Framework exceptions + logging | not started |

## Infrastructure status

The repository stack is modeled on NumSharp's. What exists today, and what is deferred until there
are algorithms to exercise:

| Capability | Status |
|------------|--------|
| Solution, multi-target projects (net8.0 / net10.0) | ✅ in place |
| Strong-naming (shared SciSharp key) + verifier gate | ✅ in place |
| Unit tests (MSTest v3) | ✅ scaffolding smoke tests |
| Live-Optuna interop (pythonnet) | ✅ harness + smoke tests |
| NuGet packaging + tag-driven release pipeline | ✅ in place |
| This DocFX site + GitHub Pages deploy | ✅ in place |
| Community health (issue/PR templates, security, funding, citation) | ✅ in place |
| Benchmarks (BenchmarkDotNet) | ✅ harness scaffold; suites grow with the ops |
| Differential-parity **corpus** oracle (à la NumSharp's committed NumPy corpus) | ⏳ deferred — the **interop suite** is the differential gate until there is an algorithm surface to snapshot |

The Optuna analog of NumSharp's committed NumPy-oracle corpus is a future addition: for now,
correctness is proven against **live** Optuna through the interop project.

# Porting Roadmap

OptunaSharp is being ported module-by-module, tracking Optuna's own layout. The core algorithm
surface is in place; the table below tracks where each module stands.

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
| `distributions` | Search-space distributions (float / int / categorical) | ✅ ported |
| `trial` | `Trial` + `FrozenTrial`, the `suggest_*` API | ✅ ported |
| `study` | `Study`, `optimize`, directions, storage glue | ✅ ported |
| `samplers` | random, grid, TPE, CMA-ES, GP, QMC, NSGA-II/III, brute-force, partial-fixed | ✅ ported |
| `pruners` | median, percentile, successive halving, Hyperband, threshold, patient, Wilcoxon, nop | ✅ ported |
| `storages` | in-memory + RDB (SQLite / EF Core) + Redis journal + gRPC proxy | ✅ ported |
| `search_space` | Search-space inference | ✅ ported |
| `importance` | Hyperparameter importance (PED-ANOVA, fANOVA, MDI, SHAP) | ✅ ported |
| `terminator` | Automated termination | ✅ ported |
| `visualization` | Optimization-history plots | 🟡 data records ported; Plotly rendering package planned |
| `exceptions` / `logging` | Framework exceptions + logging | ✅ ported |

## Infrastructure status

The repository stack is modeled on NumSharp's. What exists today, and what is deferred until there
are algorithms to exercise:

| Capability | Status |
|------------|--------|
| Solution, multi-target projects (net8.0 / net10.0) | ✅ in place |
| Strong-naming (shared SciSharp key) + verifier gate | ✅ in place |
| Unit tests (MSTest v3) | ✅ full suites (committed-oracle + behavior tests, both TFMs) |
| Live-Optuna interop (pythonnet) | ✅ differential parity harness + suites |
| NuGet packaging + tag-driven release pipeline | ✅ in place |
| This DocFX site + GitHub Pages deploy | ✅ in place |
| Community health (issue/PR templates, security, funding, citation) | ✅ in place |
| Benchmarks (BenchmarkDotNet) | ✅ harness scaffold; suites grow with the ops |
| Differential-parity **corpus** oracle (à la NumSharp's committed NumPy corpus) | ⏳ deferred — the **interop suite** is the differential gate until there is an algorithm surface to snapshot |

The Optuna analog of NumSharp's committed NumPy-oracle corpus is a future addition: for now,
correctness is proven against **live** Optuna through the interop project.

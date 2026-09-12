---
name: benchmarking
description: How OptunaSharp is benchmarked along its two orthogonal axes — SPEED and QUALITY — and how to add, run, or design either. SPEED = the BenchmarkDotNet harness at test/OptunaSharp.Benchmark (per-operation wall-clock + allocations via [MemoryDiagnoser]); today it holds only ScaffoldingBenchmarks, so add [Benchmark] methods as ops land and run with `dotnet run -c Release -f net8.0 -- --filter` (CI is build-smoke only). QUALITY = VolleyEval scenario benchmarking (design at docs/BENCHMARK_VOLLEYEVAL_PLAN.md, illustrated in docs/volleyeval-report.html): fire a volley of optimization challenges at every sampler, auto-tune each, and score a normalized compute-to-threshold PROFILE MATRIX — because there is no single best sampler, only a best sampler per regime. Covers the normalization (best−random)/(optimum−random), the two metrics (best-at-budget + trials-to-threshold), seed distributions (median+IQR), DNF/inert as first-class results, the nested-hyperopt + held-out-seed protocol, the two leaderboards (per-trial-budget + per-wall-clock), the per-sampler figure-of-merit tuple, and a ready-to-run `dotnet run` recipe that ranks the real samplers on synthetic objectives today. Use when asked to benchmark OptunaSharp, add or run a benchmark, measure a sampler's speed or quality, compare or rank samplers, reason about which sampler wins a regime, or build the VolleyEval suite.
keywords: benchmark, benchmarking, BenchmarkDotNet, perf, wall-clock, MemoryDiagnoser, sampler ranking, VolleyEval, profile matrix, compute-to-threshold, normalized score, best-at-budget, trials-to-threshold, per-regime champion, nested hyperopt, held-out validation, two leaderboards, DNF, inert, quality benchmark, speed benchmark, sampler shootout
keywords-sparse: benchmark optunasharp, add a benchmark, run benchmarkdotnet, rank samplers, compare samplers, volleyeval, profile matrix, compute-to-threshold, how fast is the sampler, which sampler is best, quality vs speed
---

# benchmarking — OptunaSharp's two axes: speed and quality

OptunaSharp benchmarks along **two orthogonal axes**, kept in different places because they answer
different questions. Conflating them is the single most common benchmarking mistake here.

| axis | question | where | unit | status |
|------|----------|-------|------|--------|
| **Speed** | how *fast* is one operation? | `test/OptunaSharp.Benchmark` (BenchmarkDotNet) | ns/op + bytes allocated | **real harness, scaffolding benchmarks only** |
| **Quality** | how *good* are the points a sampler proposes, per unit of budget? | VolleyEval — `docs/BENCHMARK_VOLLEYEVAL_PLAN.md` | normalized compute-to-threshold ∈ [0,1] | **design only; prototype it with `dotnet run` today** |

The headline lesson that makes the quality axis exist: **there is no single best sampler, only a best
sampler per regime.** In one investigation the ranking *inverted* with the problem — QMC beat TPE at
low budget on categoricals, LM-CMA beat NSGA-III 12–102× on 128-D continuous, and CMA-ES was
bit-identical to random on categoricals. Any benchmark that reports one number is misleading; the
quality axis reports a **profile matrix** (sampler × challenge) instead. See the rendered walkthrough
at `docs/volleyeval-report.html`.

## 0. What actually exists (don't benchmark fiction)

- **Samplers ported** (`src/OptunaSharp.Core/Samplers/`, the surface you can benchmark today):
  `RandomSampler`, `GridSampler`, `BruteForceSampler`, `PartialFixedSampler`, `QMCSampler`,
  `TPESampler`, `GPSampler`, `CmaEsSampler`, `LmCmaSampler`, `NSGAIISampler`, `NSGAIIISampler`.
  There is **no** `SepCmaSampler` / `CmaWithMarginSampler` / native `pycma` / `botorch` sampler — the
  VolleyEval plan's contender list includes aspirational + Python-integration entries; rank only the
  real types above.
- **Speed harness is real** but holds `ScaffoldingBenchmarks.cs` (a NumSharp `arange+add` and a
  version-length no-op) — a template proving the harness compiles and runs, not a meaningful number.
- **VolleyEval is not built.** No `experiments/`, no `OptunaSharp.Benchmark.VolleyEval` project, no
  committed B1–B12 challenges yet. The plan's "B5/B6 already exist as scratch experiments" refers to a
  prior ad-hoc session, not code in this tree. To do quality benchmarking *now*, use the `dotnet run`
  recipe in §2.
- **The CLAUDE.md "scaffolding / no algorithms ported" header is stale** — 11 samplers, GP, NSGA
  linalg, storages, importance, and the interop parity gate all exist. Trust the `src/` tree and the
  `testing` skill over that header.

Core API you will use (verified public surface):

```csharp
using OptunaSharp;
var study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: new TPESampler(seed: 0));
study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 200);
double best = study.BestValue;   // minimize by default; also study.Trials, study.BestParams
```

---

## 1. Speed axis — BenchmarkDotNet (`test/OptunaSharp.Benchmark`)

The harness is a plain console: `Program.cs` runs
`BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args)`; benchmark classes carry
`[MemoryDiagnoser]`, `[Params(...)]`, and `[Benchmark]` methods. Mirror `ScaffoldingBenchmarks.cs`.

### Add a benchmark
Drop a class into `test/OptunaSharp.Benchmark/`. Measure a *whole meaningful operation* (a study run,
a single `sampler` ask, an eigendecomposition-heavy step), never a micro-slice that JIT elides.

```csharp
using BenchmarkDotNet.Attributes;
using OptunaSharp;

[MemoryDiagnoser]
public class SamplerBenchmarks
{
    [Params(50, 200)] public int NTrials;

    [Benchmark(Baseline = true)]
    public double Random_Sphere8() => RunSphere(new RandomSampler(0), NTrials);

    [Benchmark]
    public double Tpe_Sphere8() => RunSphere(new TPESampler(seed: 0), NTrials);

    [Benchmark]
    public double CmaEs_Sphere8() => RunSphere(new CmaEsSampler(seed: 0), NTrials);

    static double RunSphere(BaseSampler sampler, int nTrials)
    {
        var study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler);
        study.Optimize(t => { double s = 0; for (int i = 0; i < 8; i++) { var x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; } return s; },
                       nTrials: nTrials);
        return study.BestValue;
    }
}
```

### Run it
```bash
# Release is mandatory — BenchmarkDotNet refuses Debug. Pick one TFM.
dotnet run --project test/OptunaSharp.Benchmark/OptunaSharp.Benchmark.csproj -c Release -f net8.0 -- --filter '*Sampler*'
dotnet run --project test/OptunaSharp.Benchmark/OptunaSharp.Benchmark.csproj -c Release -f net8.0 -- --filter '*' --job short   # quick pass
```
Results land in `BenchmarkDotNet.Artifacts/` (markdown + csv). `--job short` trades statistical rigor
for speed while iterating; drop it for a number you'll quote.

### Gotchas
- **Never** run this under `dotnet test` — it is a console exe, `IsPackable=false`, not a test suite.
- Compare TFMs deliberately: net8.0 vs net10.0 can differ; the `Sampler` allocation profile is the
  interesting axis (`[MemoryDiagnoser]` reports Gen0/1/2 + bytes) since NSGA/CMA/GP allocate NDArrays.
- **CI is build-smoke only** (`.github/workflows/benchmark.yml`): every push compiles the harness; a
  full `--job short` run is `workflow_dispatch`-manual — a perf suite isn't meaningful until it holds
  real `[Benchmark]`s. Adding real ones is the point of this skill.
- Comparing to Python? Speed-vs-Optuna is fair game (wall-clock per `sampler.step`), but keep it out
  of the *quality* axis — parity of *values* is the `testing` skill's job, not the benchmark's.
- `dotnet` caveat on this machine: taskkill isn't available; if a build wedges on a locked DLL, pause
  for the user rather than force-killing.

---

## 2. Quality axis — VolleyEval scenario benchmarking

The quality benchmark ranks samplers by **compute-to-threshold on a catalog of challenges chosen to
span "difficulty space,"** then normalizes so challenges are comparable. Full design in
`docs/BENCHMARK_VOLLEYEVAL_PLAN.md`; the rules that matter operationally:

### The measurement rules (non-negotiable)
1. **Two metrics per challenge, never one.** *Best-at-budget* (final value after a fixed trial budget)
   AND *trials-to-threshold* (trials to first reach 50/90/99% of the optimum-above-random band).
2. **Normalize to a fixed band.** `score = (best − random) / (optimum − random)` ∈ [0,1]. `0` = no
   better than random; `1` = optimal. Synthetic challenges carry exact optima; real ones use
   best-ever-seen as a *provisional* proxy and are marked so.
3. **Report a seed distribution** (median + IQR), never a bare mean — seed variance is large
   (a single-seed 13140 validated to a 12277–13140 spread). Seed everything; RNG must be reproducible.
4. **DNF and inert are first-class results.** A sampler that can't finish the budget in sane
   wall-clock is `DNF(infeasible)`; one that abstains on a parameter type (native CMA on categoricals →
   random) is `≡random(inert)`. Record them; they are signal, not a gap.
5. **Two leaderboards.** *Per-trial-budget* (all samplers get N trials → algorithmic efficiency) AND
   *per-wall-clock* (all get T seconds → practical efficiency). Per-trial cost spans 20–800×; a
   cost-blind board is dishonest.

### The hyperoptimization protocol (each challenge tunes the sampler it faces)
Every round is a **nested study**: an outer study (TPE) searches the sampler's own knob space; the
inner objective runs the challenge at budget and returns the aggregated best; the outer winner is
**re-validated on held-out seeds** (mandatory — guards seed-overfit); the validated knobs are filed as
a recipe keyed by `(sampler, challenge, budget)`. Tuning is **budget-specific** (small populations won
*only* at low budget), so report per-budget.

### The measurable unit per model → a tuple, not one scalar
Atomic unit: **normalized score per (sampler × challenge)**. Roll a sampler's row up to its figure of
merit: **mean** (robustness), **regimes won** (rank-tournament / how often best), **peak** (does it own
any regime), **coverage** (did it compete, or DNF/abstain) — each also **per-wall-clock**. Report the
tuple; do not collapse to one headline number (that hides the regime inversion). The highest-mean, the
most-wins, and the highest-peak sampler are routinely three different samplers.

### Prototype a quality benchmark TODAY
No VolleyEval project is needed to start. A .NET 10 file-based app (anywhere under the repo root, so it
inherits `Directory.Build.props` and restores NumSharp) ranks the real samplers on synthetic
minimizers with known optimum `0`:

```csharp
#:project src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
using OptunaSharp;

const int D = 16, Budget = 300, Seeds = 5;

static double Sphere(IReadOnlyList<double> x)      { double s = 0; foreach (var v in x) s += v * v; return s; }
static double Rosenbrock(IReadOnlyList<double> x)  { double s = 0; for (int i = 0; i < x.Count - 1; i++) { var a = 1 - x[i]; var b = x[i + 1] - x[i] * x[i]; s += a * a + 100 * b * b; } return s; }
static double Rastrigin(IReadOnlyList<double> x)   { double s = 10 * x.Count; foreach (var v in x) s += v * v - 10 * Math.Cos(2 * Math.PI * v); return s; }

var challenges = new (string name, Func<IReadOnlyList<double>, double> f, double lo, double hi)[]
{
    ("sphere",     Sphere,     -5, 5),
    ("rosenbrock", Rosenbrock, -5, 5),
    ("rastrigin",  Rastrigin,  -5.12, 5.12),
};
var contenders = new (string name, Func<int, BaseSampler> make)[]
{
    ("random", s => new RandomSampler(s)),
    ("qmc",    s => new QMCSampler(seed: s)),
    ("tpe",    s => new TPESampler(seed: s)),
    ("cmaes",  s => new CmaEsSampler(seed: s)),
    ("lmcma",  s => new LmCmaSampler(seed: s)),
    ("nsga2",  s => new NSGAIISampler()),          // GA: its knobs (populationSize, crossoverProb) are its tuning space
};

double BestAtBudget(BaseSampler sampler, Func<IReadOnlyList<double>, double> f, double lo, double hi)
{
    var study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler);   // minimize by default
    study.Optimize(t => { var x = new double[D]; for (int i = 0; i < D; i++) x[i] = t.SuggestFloat($"x{i}", lo, hi); return f(x); }, nTrials: Budget);
    return study.BestValue;
}

Console.WriteLine($"{"sampler",-8} " + string.Join(" ", challenges.Select(c => $"{c.name,12}")));
foreach (var (name, make) in contenders)
{
    var cells = challenges.Select(c =>
    {
        // random defines the 0 of the band; optimum is 0 → score = 1 − best/randomBest, seed-median.
        double randMed = Median(Enumerable.Range(0, Seeds).Select(s => BestAtBudget(new RandomSampler(s), c.f, c.lo, c.hi)));
        double med     = Median(Enumerable.Range(0, Seeds).Select(s => BestAtBudget(make(s), c.f, c.lo, c.hi)));
        return Math.Clamp(1.0 - med / randMed, 0, 1);
    }).ToArray();
    Console.WriteLine($"{name,-8} " + string.Join(" ", cells.Select(v => $"{v,12:F3}")));
}

static double Median(IEnumerable<double> xs) { var a = xs.OrderBy(v => v).ToArray(); return a[a.Length / 2]; }
```
```bash
dotnet run scratch/quality_probe.cs      # ~a minute; prints a mini profile matrix (score ∈ [0,1])
```
Extensions toward the real suite, in priority order: (a) add **trials-to-threshold** by iterating
`study.Trials` for the first trial whose running-min crosses each band; (b) add an **ill-conditioned
ellipsoid** and a **categorical reward** challenge (maximize → negate, or pass a maximize direction);
(c) wrap each cell in the **nested-hyperopt** loop (an outer `Optuna.CreateStudy` whose objective is
`BestAtBudget`, searching the sampler's ctor knobs) with **held-out-seed** validation; (d) add the
**per-wall-clock** twin by budgeting on a `Stopwatch` instead of `nTrials`.

> Member names (`BestValue`, `Trials`, `FrozenTrial.Value`/`Number`) are the public `Study` surface —
> if a build errors, confirm against `src/OptunaSharp.Core/{Study,Trial,FrozenTrial}.cs`. Give every
> sampler that accepts a `seed` one, for reproducibility.

### Reuse existing OptunaSharp surfaces (don't hand-roll)
- **`OptunaSharp.Importance` (fANOVA / PED-ANOVA)** — after a hyperopt, report *which* knob explained
  the sampler's variance ("`populationSize` explains 80% of NSGA-III's spread"). Turns tuning into
  insight.
- **`WilcoxonPruner`'s signed-rank test** — reuse for pairwise significance between samplers ("is
  tuned NSGA-III *significantly* above the runner-up on this challenge, or within seed noise?").
- **`OptunaSharp.Storage.Sqlite`** — persist every round so results accumulate across sessions and
  become the baseline for a regression diff.
- The **nested-study pattern is just an OptunaSharp study whose objective runs an OptunaSharp study** —
  "optimize the optimizer," realized inside the library.

---

## 3. Honesty rules (both axes)

- **Run real Optuna first, observe, then compare** — the repo-wide invariant. Under a fixed seed,
  OptunaSharp reproduces Optuna bit-for-bit (RNG-deterministic samplers) or to erf-tolerance (TPE/GP);
  a quality benchmark whose numbers don't match Python at matched seed has a parity bug, not a result.
- **Seed everything; report distributions, not means.** A single-seed win is a hypothesis until it
  survives held-out seeds.
- **Mark provisional vs measured.** Synthetic optima are exact; real-surface optima are best-seen
  proxies — say so. The `docs/volleyeval-report.html` matrix distinguishes ● measured (anchored to
  real evidence) from ○ hypothesis cells; keep that discipline in any results you publish.
- **Speed ≠ quality.** A sampler can win batches-to-threshold and lose wall-clock-to-threshold because
  its per-step cost is higher (CMA/GP eigendecomps). Report both, never silently pick the flattering
  one.

## 4. Definition of done

**Speed benchmark:** a `[Benchmark]` class under `test/OptunaSharp.Benchmark/` measuring a real
operation, `[MemoryDiagnoser]` on, runs green via `dotnet run -c Release … --filter`, results committed
or quoted from `BenchmarkDotNet.Artifacts/`. Commit `bench(perf): …` (or `test(bench): …`).

**Quality benchmark:** ranks only the real samplers (§0), every cell normalized to [0,1] with a stated
optimum + random baseline, reported as a seed distribution, DNF/inert recorded not dropped, both
leaderboards where per-step cost differs, and any surprising single-seed finding validated on held-out
seeds. If it grows into a standing suite, that is the `OptunaSharp.Benchmark.VolleyEval` project the
plan specifies (sibling to the perf `OptunaSharp.Benchmark`).

## 5. See also
- `docs/BENCHMARK_VOLLEYEVAL_PLAN.md` — the full quality-suite design (taxonomy, challenges B1–B12,
  architecture, phased rollout).
- `docs/volleyeval-report.html` — the rendered, interactive profile-matrix walkthrough of the design.
- **`testing` skill** — parity framework, RNG discipline, running the suites (the correctness gate the
  benchmark's numbers rest on).
- **`interop-test-porting` skill** — capturing real-world Optuna usage as parity fixtures (a source of
  realistic quality challenges, e.g. the B12 real-HPO-surrogate).

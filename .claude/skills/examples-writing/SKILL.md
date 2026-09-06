---
name: examples-writing
description: Author runnable OptunaSharp examples under examples/ — .NET 10 file-based apps that showcase real usage. Covers the two kinds (ports of top-starred gists vs. feature showcases), the file-based-app skeleton and prerequisites, the real-vs-surrogate objective policy and numeric gotchas, MANDATORY source linking (verified URLs), the lowercase in-code comment style, optional byte-for-byte parity pairing in the interop suite, updating examples/README.md, and the path-limited commit. Use when adding or editing an example, deciding an example's objective, or answering "what examples should we add".
keywords: write example, add example, examples folder, optunasharp example, file-based app, showcase example, gist port example, example objective, surrogate objective
keywords-sparse: how do I write an OptunaSharp example, add a new example, what examples to add, examples/ conventions, link the gist, lowercase comments, surrogate vs real objective
---

# examples-writing — authoring OptunaSharp `examples/`

Examples are **runnable .NET 10 file-based apps** (the global `dotnet-run-script` skill) in `examples/`, each
showing one slice of real OptunaSharp usage, launched with `dotnet run examples/<Name>.cs`. They are the
project's shop window: correct, self-contained, readable, and — where the objective allows —
byte-for-byte checked against live Optuna. Two kinds:

- **Gist ports** — the most-starred real-world Optuna usage. Find/rank them with the
  [`interop-test-porting`](../interop-test-porting/SKILL.md) skill (`harvest.sh gists-ranked`), then port
  faithfully. Current: `FFmpegX264Tuning` (sile 25★), `XGBoostHpo` (bexgboost 16★), `ParamRepeatPruner`
  (aisu-wata0 6★).
- **Feature showcases** — a canonical OptunaSharp/Optuna capability that has no example. Current:
  `MedianPrunedTraining`, `MultiObjectivePareto`, `HyperparameterImportance`, `AutomaticTermination`.

## Prerequisites (what a build needs)

Examples `#:project` the Core, which pulls **NumSharp + NumSharp.Interop.OpenBLAS from NuGet** — nothing
to set up. The OpenBLAS package ships the LAPACK natives (copied to the run output via `buildTransitive`),
so even examples that touch **np.linalg** (GPSampler, CmaEsSampler, the GP-backed Terminator, NSGA-III's
hyperplane solve) just work at runtime. No Python is needed to *run* an example (Python is only for the
parity test that backs it).

## File skeleton

```csharp
#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// <Title> — one line on what this shows.
//
// source: <verified gist/repo/docs URL(s)>          ← MANDATORY (see "Linking")
//
// <why the objective is real or a surrogate; what feature it exercises; the run line>
//
//   Run:  dotnet run examples/<Name>.cs

using OptunaSharp;              // Study, Trial, Optuna, StudyDirection, TrialState, TrialPruned
using OptunaSharp.Samplers;     // RandomSampler, TPESampler, NSGAIISampler, ...
using OptunaSharp.Storages;     // InMemoryStorage

var study = Optuna.CreateStudy(/* ... */);
study.Optimize(Objective, nTrials: N);
// ... print results ...

static double Objective(Trial t) { /* ... */ }   // top-level statements FIRST, then methods/types
```

**File-based-app rule:** top-level statements come first; `static` methods and any type declarations
come **after** them (types last). The objective is a `static` method after the top-level code.

## Objective policy — real vs. surrogate

A sampler's trajectory is fixed by `(seed, search space, value sequence)` alone, so the objective's
*value* need not be produced by the real workload. Choose:

- **Real objective** when it's self-contained and runs without a foreign ML runtime in-process — shell
  out (ffmpeg via `Process`), a pure computation, a real check (`ParamRepeatPruner`). Best fidelity.
- **Surrogate (deterministic closed-form)** when the real objective needs something C# can't run
  in-process (XGBoost/LightGBM training, a neural net). **Keep the gist's exact search space**; replace
  only the value. The real-ML run then belongs in `OptunaSharp.Tests.PackageParity` (replay), not here.

**Numeric gotchas that keep an example honest / parity-exact:**
- Scalar power: `Math.Pow(x, 2.0)`, never `x*x` — matches Python `x ** 2` bit-for-bit.
- Format floats for external tools with `CultureInfo.InvariantCulture` (never a locale comma).
- `SuggestInt` returns `long`; `SuggestFloat` returns `double`; `SuggestCategorical(name, int[])` →
  `long`, `(name, string[])` → `string`, `(name, bool[])` → `bool`.
- Log-scale search: pass `log: true` to `SuggestFloat`/`SuggestInt` for span-many-orders params.
- For a clean **importance** demo use `RandomSampler` (uniform coverage) + `FanovaImportanceEvaluator`
  (variance decomposition) — TPE concentrates and PedAnova blurs the ranking.

## Linking the source (MANDATORY)

Every example header carries a `source:` line. **Verify the URL before writing it** — never guess a
path:
- Gist: the gist URL (you already have it from `harvest.sh`).
- Repo pattern: `env -u GH_TOKEN -u GITHUB_TOKEN gh api repos/<owner>/<repo>` (200 = exists); for a file,
  `.../contents/<path>`.
- Optuna docs / examples: `curl -s -o /dev/null -w "%{http_code}" -L <url>` must be `200`. Good anchors:
  `optuna.readthedocs.io/en/stable/reference/<x>.html`, `.../tutorial/...`,
  `github.com/optuna/optuna-examples/blob/main/<path>`.

## Comment style

Add generous **lowercase-first** inline comments — `// like this` — explaining the study options, every
`suggest_*`, the surrogate math, and the feature's mechanics. Trailing comments aligned on the
interesting lines read best. (Proper nouns mid-sentence keep their case; it's the comment's leading word
that's lowercase.) The header doc block stays sentence-case prose.

## Byte-for-byte parity (pair it when the objective is deterministic)

The example itself is C#-only; the byte-for-byte check lives in the **interop suite** and is what proves
the port. Add/extend a parity test when the objective is deterministic (per the
[`testing`](../testing/SKILL.md) skill §5):
- Build the C# study + run the identical study as a Python string via `PyExec`; `StudySnapshot.Capture`
  vs `PyStudyState`; `StudySnapshot.AssertParity`.
- **Bit-exact** for RNG-deterministic samplers (Random, Grid, BruteForce, QMC, CMA-ES, NSGA-II/III,
  PartialFixed). **Tolerance** for TPE/GP (erf path). **Exclude TPE over integer spaces** (NumPy
  simd-sort tie order isn't byte-comparable).
- Existing backing: `GistExampleParityTests` (FFmpeg + XGBoost spaces), `ParamRepeatPrunerParityTests`.
  The showcase *features* are already covered (`PrunerDecisionParityTests`, `NsgaiiSamplerParityTests`,
  `ImportanceParityTests`, `TerminatorParityTests`) even where the specific showcase objective isn't.
- Real-ML objectives (LightGBM via `OptunaSharp.Integration.LightGbm`, xgboost, sklearn) → a
  `gen_oracle.py` scenario + `RealPackageParityTests` row in PackageParity (Ask/Tell replay).

## Then: README + verify + commit

1. **README** — add a row to the right table in `examples/README.md` (**Ported from real-world gists**
   or **Feature showcases**), with the source link, and keep the run-commands + real-vs-surrogate notes
   current.
2. **Run it** — `dotnet run examples/<Name>.cs`; confirm the output is sensible (pruned counts, a Pareto
   front, a clear importance ranking, an early stop…). If it has a parity test, run it with
   `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1` and `PYTHONNET_PYDLL=<libpython>`.
3. **Commit path-limited** — parallel sessions share the git index, so stage explicit paths and commit
   with a pathspec so you never sweep another session's staged work:
   `git add examples/<Name>.cs examples/README.md && git commit -F msg -- examples/<Name>.cs examples/README.md`.
   Build by csproj path, never the sln.

## Hard-won gotchas (from the examples built so far)

- **Terminator**: the default error evaluator (`CrossValidationErrorEvaluator`) *requires* per-trial
  `TerminatorFunctions.ReportCrossValidationScores` — without it, `ShouldTerminate` throws. For a quiet
  demo use `BestValueStagnationEvaluator` (auto-pairs a `StaticErrorEvaluator`, no CV reporting). Note:
  `optuna.terminator` is deprecated upstream (v4.9 → removed v6), so a `FutureWarning` prints.
- **Namespaces**: fully-qualify `OptunaSharp.Terminator.Terminator` / `.TerminatorCallback` (the type
  shares its name with the namespace). `OrderedDict<,>` needs `using OptunaSharp.Collections;`.
- **Multi-objective**: `directions:` (plural) + objective returns `IReadOnlyList<double>`; the Pareto
  front is `study.BestTrials`, and each trial's outputs are `t.Values` (list), not `t.Value`.
- **Callbacks**: `study.Optimize(..., callbacks: new[] { (Action<Study, FrozenTrial>)cb })`; `nTrials` is
  a cap a callback (e.g. the terminator) can stop short.
- **ffmpeg SSIM**: encode with `-x264-params "...:ssim=1"` and parse `SSIM Mean Y:\s*([0-9.]+)` from
  stderr; a failed combo → score 0 (worst). YouTube blocks yt-dlp downloads (`403`) without cookies —
  always provide a fallback (an ffmpeg `testsrc2` clip) so the example still runs.

## Current coverage & candidate next examples

Have: FFmpeg (SHA pruner), XGBoost (HPO grid), ParamRepeatPruner (custom pruner), Median-prune (
MedianPruner), Multi-objective (NSGA-II), Importance (fANOVA/PedAnova), Terminator.

Still uncovered (good next examples; ✅ = comes with byte-for-byte): **GridSampler** all-categorical
(SyntheticTumors 378★) ✅; **real LightGBM HPO** via `Integration.LightGbm` (→ PackageParity replay) ✅;
**CMA-ES** continuous ✅; **resumable JournalStorage** (`load_if_exists`, TabSurvey 281★) ✅;
**QMC / NSGA-III / PartialFixed / BruteForce** ✅; **constrained optimization** (`constraints_func`) ✅;
**warm-start** (`enqueue_trial`) ✅; **GP / Bayesian optimization** (⚠️ tolerance, erf path).

## Definition of Done

Runs green with sensible output · header carries a **verified** `source:` link · lowercase in-code
comments throughout · `examples/README.md` updated · a byte-for-byte parity test where the objective is
deterministic (or a note why not) · committed **path-limited** with a `feat(examples):`/`docs(examples):`
message.

## Copy-paste template

```csharp
#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// <Title> — <one line>.
//
// source: <verified URL>
//
// <real vs surrogate note; feature exercised>
//
//   Run:  dotnet run examples/<Name>.cs

using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),        // in-memory — nothing persisted
    sampler: new RandomSampler(seed: 0),   // seeded, so the run reproduces
    direction: StudyDirection.Minimize,
    studyName: "<name>");

study.Optimize(Objective, nTrials: 30);

Console.WriteLine($"Best value: {study.BestValue}");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

// <what the objective is; keep the real search space, surrogate the value if needed>
static double Objective(Trial t)
{
    double x = t.SuggestFloat("x", -10, 10);   // <comment each suggest>
    return Math.Pow(x - 2, 2.0);               // Math.Pow, not x*x, for Python x**2 parity
}
```

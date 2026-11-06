# Plan — ML.NET importance backends (Forests / Boosters) + LightGBM integration

> **Audience:** the next engineer picking up the Phase 5.1 forest work on the branch
> `feat/phase-5.1-importance-forests` (since merged into `master`).
> **Governs:** how the ML.NET-backed importance evaluators are split across packages/namespaces,
> and how LightGBM is scoped. Read `docs/PORTING_PLAN.md` §5.1 first for the phase context.

> **⚠️ Update (2026-09-05) — packages consolidated.** The two FastTree-runtime importance packages
> described below — `OptunaSharp.Importance.Forests.ML.NET` (FastForest) and
> `OptunaSharp.Importance.Booster.ML.NET` (FastTree/FastTreeTweedie) — have since been **merged into a
> single package `OptunaSharp.Integration.ML.NET`.** They shared the exact same native dependency
> (`Microsoft.ML.FastTree`), so one package now hosts both (the faithful FastForest backend + the
> experimental boosters). **Public namespaces are unchanged** (`OptunaSharp.Importance.Forests` /
> `OptunaSharp.Importance.Booster`) — only the assembly / `PackageId` changed. **LightGBM stays its
> own package** (`OptunaSharp.Integration.LightGbm`): distinct heavy native runtime
> (`Microsoft.ML.LightGbm`) + its own Optuna identity (`optuna.integration.lightgbm`), per the
> "one package per heavy native dep" rule. The §1 taxonomy table reflects the merge; the WI-1…WI-4
> work items further down are the original build history and still name the pre-merge packages.

---

## ✅ STATUS — COMPLETE (all work items landed on `master`)

Done directly on `master` (not the worktree named at the top — the Phase 5.1 forest work had already
merged), each work item green (build ×2 TFMs + tests + strong-named pack) before the next:

- **WI-0** — prerequisite satisfied (NumSharp + NumSharp.Interop.OpenBLAS restore from NuGet; plain
  `dotnet build` by csproj path).
- **WI-1** — `refactor(importance)` **`0403972`**: FastForest package rename was already committed
  (`647e2ec`); its evaluators moved to ns `OptunaSharp.Importance.Forests`; call sites + Core
  doc-comments + FastForest README updated.
- **WI-2** — same commit **`0403972`**: `MlNetTreeEnsembleAdapter` extracted to the shared linked source
  `src/_shared/MlNetTreeEnsembleAdapter.cs` (ns `OptunaSharp.Importance.MLNet`); `FastForestBackend.Fit`
  now `ensemble.Trees.Select(MlNetTreeEnsembleAdapter.ToSklearnLikeTree)`.
- **WI-3** — `feat(importance)` **`599bf3f`**: new package `OptunaSharp.Importance.Booster.ML.NET`
  (FastTree + FastTreeTweedie, ns `OptunaSharp.Importance.Booster`), experimental/off-label; unit suite
  `OptunaSharp.Tests.Importance.Booster` (10 ×2 TFMs) + interop ranking parity (8 vs live Optuna).
- **WI-4** — `feat(integration)` **`c7d9af9`**: new package `OptunaSharp.Integration.LightGbm` (LightGBM
  importance now, tuner scoped for Phase 5.7), experimental/off-label; unit suite
  `OptunaSharp.Tests.Integration.LightGbm` (10 ×2 TFMs) + interop ranking parity (4 vs live Optuna).

All three satellite packages pack strong-named (`verify_strong_name.cs` OK, token `cc7b13ffcd2ddd51`);
Core `InternalsVisibleTo` grants added for both new backend assemblies + their test projects; all six
new/renamed projects are in the `.sln`. Live interop ran green against Optuna 5.0.0.dev + scikit-learn
1.8.0. Remaining follow-up: the LightGBM **tuner** (Phase 5.7) and CI wiring (deferred, as with 4.3/4.5).

---

## 0. TL;DR of the decision

Optuna's importance module (`optuna.importance`) uses **only `sklearn.ensemble.RandomForestRegressor`**
— fANOVA + MDI — plus the pure-numpy PED-ANOVA. **There is no boosting anywhere in Optuna's
importance.** So:

- The **canonical** `FanovaImportanceEvaluator` / `MeanDecreaseImpurityImportanceEvaluator` are the
  pure-managed, **bit-exact scikit-learn CART port in Core** (Tier N, value-parity vs live Optuna).
  Nothing below changes them.
- The **ML.NET FastForest** backend is the only *faithful* ML.NET analog (same model as Optuna, ranking
  parity) → package `OptunaSharp.Integration.ML.NET`.
- **Boosters** (FastTree / FastTreeTweedie) have **no Optuna basis** and are *off-label* for fANOVA/MDI
  (their trees fit residuals, so the importances are a different, non-standard quantity). They ship as
  clearly-labelled **experimental extensions** alongside FastForest in `OptunaSharp.Integration.ML.NET`.
- **LightGBM's** real Optuna identity is a **tuner** (`optuna.integration.lightgbm` — our Phase 5.7),
  not importance. It goes in **`OptunaSharp.Integration.LightGbm`**, which will host the tuner *and* a
  LightGBM booster-importance backend (one package per heavy native dep).

### What "boosters" are (for the record)
- **Bagging / random forest** (FastForest): independent trees, each fits the target `y`, ensemble =
  **average**. Variance reduction. This is what fANOVA/MDI were designed for.
- **Gradient boosting** (FastTree = MART, FastTreeTweedie = Tweedie-loss MART, LightGBM = histogram
  GBM): sequential trees, each fits the **residual** of the ensemble-so-far, ensemble = **weighted
  sum** with a learning rate. Bias reduction. Trees model residuals, not the objective → importance
  over them means something different (hence "experimental" here).

---

## 1. Final taxonomy (packages / namespaces / csproj)

| Concern | Package = AssemblyName = PackageId | Public evaluator namespace | Internal namespace | ML.NET dep | Optuna basis |
|---|---|---|---|---|---|
| Canonical fANOVA/MDI (native CART) | `OptunaSharp` (Core) | `OptunaSharp.Importance` | `OptunaSharp.Importance.Forests.Internal[.Cart]` | none (NumSharp) | ✅ RandomForest, **value-exact** |
| FastForest backend | `OptunaSharp.Integration.ML.NET` | `OptunaSharp.Importance.Forests` | `OptunaSharp.Importance.Forests.MLNet` | `Microsoft.ML` + `Microsoft.ML.FastTree` | ~ same model, **ranking** parity |
| Boosters: FastTree, FastTreeTweedie | `OptunaSharp.Integration.ML.NET` *(same pkg — merged)* | `OptunaSharp.Importance.Booster` | `OptunaSharp.Importance.Booster` | `Microsoft.ML` + `Microsoft.ML.FastTree` | ⚠ **none — experimental** |
| LightGBM: importance now, tuner (5.7) later | `OptunaSharp.Integration.LightGbm` | `OptunaSharp.Integration.LightGbm` | `OptunaSharp.Integration.LightGbm` | `Microsoft.ML` + `Microsoft.ML.LightGbm` | ⚠ importance off-label; **tuner = `optuna.integration.lightgbm`** |

**Namespace rule (apply consistently):** `OptunaSharp.Importance` holds only the **Optuna-faithful
canonical** evaluators (Core). Every **alternative backend** lives in **its own child namespace**
(`…Importance.Forests`, `…Importance.Booster`, `…Integration.LightGbm`) so no two assemblies share a
namespace and the "which is the real one" question is answered by the name. Type names stay
self-describing: `FastForestFanovaImportanceEvaluator`, `FastTreeFanovaImportanceEvaluator`,
`FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator`, `LightGbmFanovaImportanceEvaluator`, …

> **Note:** WI-1 below **moves the FastForest evaluators from `OptunaSharp.Importance` to
> `OptunaSharp.Importance.Forests`** to satisfy this rule (they currently sit in `OptunaSharp.Importance`).

---

## 2. Shared architecture — reuse Core, don't duplicate the algorithm

The importance **algorithm** already lives in Core and is backend-agnostic (all `internal`, in
`OptunaSharp.Importance.Forests.Internal`):

- `SklearnLikeTree` — the sklearn flat-node tree (structure + per-node `Impurity` / `WeightedNNodeSamples`).
- `FanovaTree`, `Fanova` — the fANOVA marginal-variance walk.
- `Mdi` — `compute_feature_importances` + the forest aggregation.
- `ForestImportance.EvaluateFanova/EvaluateMdi(study, params, target, ForestFit fit)` — the whole
  evaluator body (float32-X cast, transform, folding, sort), parameterised by a **`ForestFit`
  delegate** (`SklearnLikeTree[] Fit(float[,] x, double[] y, int nFeatures)`).

**So every backend is just: `trainer.Fit → RegressionTreeEnsemble → adapt → SklearnLikeTree[] →
ForestImportance`.** A backend package writes ~2 thin evaluator classes + one delegate lambda.

### The shared ML.NET adapter (extract this — WI-2)
`RegressionTreeBase` is the common shape of **every** ML.NET tree ensemble — verified: FastForest,
FastTree, FastTreeTweedie **and** LightGBM all expose `TrainedTreeEnsemble : RegressionTreeEnsemble`
with `LeftChild / RightChild / NumericalSplitFeatureIndexes / NumericalSplitThresholds / LeafValues /
NumberOfNodes / NumberOfLeaves`. The adapter (currently inline in `FastForestBackend.Adapt`) does:
pre-order DFS → sklearn-flat nodes (leaf encoding `~child`), then routes the training set to fill
`Impurity` (node variance) + `WeightedNNodeSamples` (count).

**Extract it to a shared `MlNetTreeEnsembleAdapter`** used by all three ML.NET packages. It needs
`Microsoft.ML.FastTree` (for `RegressionTreeBase`) + Core-friend access to `SklearnLikeTree`.
**Placement — recommended:** a **linked shared source file** compiled into each package, e.g.
`src/_shared/MlNetTreeEnsembleAdapter.cs` referenced by
`<Compile Include="..\_shared\MlNetTreeEnsembleAdapter.cs" Link="Internal\MlNetTreeEnsembleAdapter.cs" />`.
This avoids both a new runtime package and awkward cross-sibling references (a "booster depends on
forests" edge). Alternative: a tiny `OptunaSharp.Importance.MLNet.Common` package — heavier, only if a
shared source file feels wrong. **Do not** put it in Core (it references ML.NET; Core stays pure managed).

---

## 3. Work items

Order matters; each WI ends green (build ×2 TFMs + tests + pack) before the next.

### WI-0 — unblock the build (prerequisite)
- NumSharp + NumSharp.Interop.OpenBLAS are **NuGet packages** (`$(NumSharpVersion)` in
  `Directory.Build.props`; see the updated `.claude/CLAUDE.md` §"The NumSharp reference"). A plain
  restore pulls them from nuget.org — no submodule init, no native staging.
- Build/test by **csproj path** — plain `dotnet build`/`dotnet test`, no weaver escape hatches needed.
- **`OptunaSharp.Tests` will not build on a clean worktree** — the Core `Artifacts/` *source* is
  gitignored (`artifacts/` matches case-insensitively on Windows) and was never committed. Put all
  new tests in the **satellite test projects** (which are Core friends), never in `OptunaSharp.Tests`.
  See `docs/PORTING_PLAN.md` and the `.gitignore` line 24. (A separate fix — un-ignore + commit those
  4 files — is out of scope here; a concurrent session owns artifacts.)

### WI-1 — finalize `OptunaSharp.Importance.Forests.ML.NET` (FastForest)
- Complete the **already-staged** directory/csproj rename `OptunaSharp.Importance.Forests` →
  `OptunaSharp.Importance.Forests.ML.NET` (git status shows the renames + the updated
  `AssemblyName`/`PackageId`, Core `InternalsVisibleTo`, both `ProjectReference`s, and the `.sln`
  entry). Validate build + tests, then commit as a small `refactor(importance)` commit.
- **Move the FastForest evaluators to namespace `OptunaSharp.Importance.Forests`** (per §1 rule);
  update the interop + satellite test `using`s.

### WI-2 — extract `MlNetTreeEnsembleAdapter`
- Pull the DFS-adapt + routed-statistics code out of `FastForestBackend` into the shared linked
  source (`src/_shared/MlNetTreeEnsembleAdapter.cs`, namespace e.g. `OptunaSharp.Importance.MLNet`).
- `FastForestBackend.Fit` becomes: build `MLContext`/options → `FastForest().Fit` →
  `ensemble.Trees.Select(t => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(t, nFeatures, x, y))`.
- Link the shared file into `OptunaSharp.Importance.Forests.ML.NET`.

### WI-3 — new package `OptunaSharp.Importance.Booster.ML.NET` (FastTree + FastTreeTweedie)
- New `src/OptunaSharp.Importance.Booster.ML.NET/` package (csproj checklist §4). Deps:
  `Microsoft.ML` + `Microsoft.ML.FastTree` (**no new native dep** — same package as FastForest).
- Link the shared adapter. Per-trainer `Fit` (only the trainer call differs from FastForest):
  `ml.Regression.Trainers.FastTree(...)` and `.FastTreeTweedie(...)`.
- Public evaluators in namespace `OptunaSharp.Importance.Booster`:
  `FastTreeFanovaImportanceEvaluator`, `FastTreeMeanDecreaseImpurityImportanceEvaluator`,
  `FastTreeTweedieFanovaImportanceEvaluator`, `FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator`
  — each a thin wrapper over `ForestImportance.Evaluate{Fanova,Mdi}` + the trainer's `Fit`.
- **XML docs must state:** experimental; gradient boosting is off-label for fANOVA/MDI (residual-fit
  importance ≠ forest importance); no value-parity with Optuna; prefer the Core canonical evaluators.
- `Assembly/Properties.cs` friend grant to the package's test project (with `PublicKey`).

### WI-4 — new package `OptunaSharp.Integration.LightGbm`
- New `src/OptunaSharp.Integration.LightGbm/` package. Deps: `Microsoft.ML` + **`Microsoft.ML.LightGbm`**
  (native LightGBM binaries per-RID — the only heavy dep in this plan).
- **Now:** the LightGBM booster-**importance** backend — `LightGbmFanovaImportanceEvaluator`,
  `LightGbmMeanDecreaseImpurityImportanceEvaluator` (namespace `OptunaSharp.Integration.LightGbm`),
  same shape as WI-3 via the shared adapter (`ml.Regression.Trainers.LightGbm(...)`), same experimental
  caveats.
- **Later (Phase 5.7):** the LightGBM **tuner** — the `optuna.integration.lightgbm.LightGBMTuner` /
  `LightGBMTunerCV` analog. This is the *Optuna-justified* half. Scope it here as a documented stub /
  follow-up; implement under Phase 5.7 (framework integrations), not in this plan unless explicitly
  pulled forward.

---

## 4. Per-package csproj checklist (mirror `OptunaSharp.Importance.Forests.ML.NET`)

- `TargetFrameworks=net8.0;net10.0`, `RootNamespace=OptunaSharp`, `AssemblyName` = `PackageId` = the
  package name from §1.
- `Description` + `PackageTags` (state the backend + "experimental, not tree-identical to scikit-learn"
  for boosters/LightGBM).
- `GenerateDocumentationFile=true`; `NoWarn=$(NoWarn);CS1591;CS1573;…`.
- `PackageReadmeFile=README.md` (+ `None Include README.md`), `PackageIcon=optunasharp-icon.png`
  (`None Include ..\..\docs\website-src\images\optunasharp-icon.png`), `PackageOutputPath=../../packages`,
  `AllowedOutputExtensionsInPackageBuildOutputFolder …;.pdb`.
- `PackageReference`s per §1. `ProjectReference` to `..\OptunaSharp.Core\OptunaSharp.Core.csproj`.
- Linked `<Compile Include>` for `_shared/MlNetTreeEnsembleAdapter.cs`.
- `Assembly/Properties.cs`: an `OptunaSharp<Name>FriendKey` class + `InternalsVisibleTo` to the
  package's **test** project (with `PublicKey=` — required, the assemblies are strong-named).
- Add the project to `SciSharp.OptunaSharp.sln` (`dotnet sln add … --solution-folder src`).

## 5. Core changes
- Add an `InternalsVisibleTo` in `src/OptunaSharp.Core/Assembly/Properties.cs` for **each new backend
  assembly** (`OptunaSharp.Importance.Booster.ML.NET`, `OptunaSharp.Integration.LightGbm`) — they reach
  Core-internal `SklearnLikeTree` / `ForestImportance` / `Fanova` / `Mdi` / `ImportanceBase` /
  `SearchSpaceTransform`. (FastForest's grant already exists as `OptunaSharp.Importance.Forests.ML.NET`.)

## 6. Testing
- **Unit** (in a satellite test project — Core friend; **not** `OptunaSharp.Tests`): determinism
  (same seed → identical), seed/`n_trees`/`max_depth` effects, normalization sums to 1 + descending
  order, MO-without-`target` throws `ValueError`, single-value dist → 0, ≤1 trial → zeros, empty
  params → empty, dominant-param ranked first. Parameterise over all backends (see the existing
  `ForestImportanceEvaluatorTests` `Backends[]` pattern) — extend it, don't fork it.
- **Interop** (`OptunaSharp.Tests.Interop`, self-skips without scikit-learn): **RANKING-only** parity
  vs live Optuna for boosters/LightGBM (dominant param ranked first on both sides). **Never assert
  value-parity for boosters** — they diverge by design. Extend `FanovaMdiImportanceParityTests`
  (it already does native value-parity + FastForest ranking).
- The `test_tree.py` bit-exact fANOVA-algorithm test stays in the (Core-friend) forests test project.
- Strong-name gate: each packed `.nupkg` must carry token `cc7b13ffcd2ddd51` (`verify_strong_name.cs`).

## 7. Definition of Done (per package)
- Builds + tests green on **net8.0 and net10.0**; packs strong-named with clean recorded deps; added to
  the sln; has a README; Core `InternalsVisibleTo` grant present.
- Boosters/LightGBM importance documented **experimental / non-Optuna** (ranking-only, off-label).
- `docs/PORTING_PLAN.md` §5.1 + package map + this file updated; memory note updated.
- LightGBM **tuner** left as a documented **Phase 5.7** follow-up (the Optuna-justified half).

## 8. Guard-rails / gotchas
- **Optuna is the source of truth:** only the Core canonical evaluators + FastForest are faithful;
  everything boosting is an OptunaSharp-only extension — keep it clearly labelled and out of
  `OptunaSharp.Importance` (the canonical namespace).
- **Concurrent sessions** advance `master` + shared files (`.sln`, `PORTING_PLAN.md`) mid-session —
  re-check git state before commit, stage by explicit path (never `git add -A`), edit shared files
  with surgical edits, build by csproj path.
- NumSharp + NumSharp.Interop.OpenBLAS come from NuGet (nuget.org), so builds are a plain restore —
  no cross-repo P2P, weaver, or native staging.

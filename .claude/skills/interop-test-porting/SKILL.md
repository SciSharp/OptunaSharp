---
name: interop-test-porting
description: Port battle-tested, real-world Optuna usage found on GitHub (the most-starred repos that `import optuna`, plus the Optuna maintainers' own gists) into byte-for-byte interop parity tests under test/OptunaSharp.Tests.Interop. Use when asked to add/port an interop or integration parity test from how real projects use Optuna, to harvest in-the-wild usage patterns, or to seed the oracle corpus from GitHub. Covers discovery (gh code-search + star enrichment; gist HTML scrape), extraction, gap-checking against existing coverage, observing live Optuna first, and writing the test in the existing StudySnapshot / OptunaInteropTestBase idiom (bit-exact vs erf-tolerance).
keywords: interop parity test, port optuna usage, github usage tests, battle-tested parity, harvest optuna gists, popular optuna repos, byte-for-byte test, integration test porting
keywords-sparse: add interop test, port a real-world optuna usage, parity test from github, harvest optuna usage, most-starred optuna repos, maintainer gists, seed oracle corpus
---

# interop-test-porting — real-world Optuna usage → byte-for-byte parity tests

Turn how real projects actually use Optuna into parity tests that assert OptunaSharp reproduces live
Optuna **byte-for-byte** (bit-exact for RNG-deterministic samplers; small tolerance for the
erf-routed ones). The differential gate lives in `test/OptunaSharp.Tests.Interop/` and runs real
CPython + Optuna in-process via pythonnet. This complements the synthetic `ParityScenario` fuzzer by
pinning the specific patterns battle-tested code depends on.

**Reference implementation to copy:** `test/OptunaSharp.Tests.Interop/BattleTestedUsageParityTests.cs`
(ported from maintainer keisuke-umezawa's optuna-dashboard fixture). Helper script for discovery:
`.claude/skills/interop-test-porting/harvest.sh`.

## Prereqs (this machine, verified)

- `gh` is authed via keyring — always call it with ambient tokens unset:
  `env -u GH_TOKEN -u GITHUB_TOKEN gh …` (the helper script does this for you).
- Live Optuna: `python` → optuna **5.0.0.dev** (must match the vendored `src/optuna` submodule; the
  interop suite installs `pip install -e src/optuna`). libpython for pythonnet:
  `PYTHONNET_PYDLL=C:/Users/ELI/.claude/python/python312.dll`.
- Core principle: **run real Optuna first, observe, then replicate.** Reproducibility under a fixed
  seed must be bit-identical (that's the whole point of the MT19937 RNG discipline).

## Workflow

### 1 — Discover real usage
```bash
SK=.claude/skills/interop-test-porting
bash $SK/harvest.sh repos 'import optuna' 100     # popular repos, ranked by stars (STARS<TAB>repo<TAB>path)
bash $SK/harvest.sh gists-ranked 25               # gists STAR-RANKED (STARS<TAB>owner/id), descending
bash $SK/harvest.sh gists 'optuna create_study'   # or: just the raw gist ids for one query (no ranking)
```
Caveats baked into the script: code search returns **no stars** (two hops — search, then enrich each
repo via `repos/{o}/{r}`), is **rate-limited 10/min**, **capped at 1000 results**, and ranked by
relevance not popularity — so it's a sample, not a census. Gists have no search API and **no stars in
REST/HTML** either — `gists-ranked` scrapes `gist.github.com/search` for a pool, then enriches each
via GraphQL `Gist.stargazerCount`. Reality check: the most-starred optuna gists are real workloads
with heavy external objectives (ffmpeg, xgboost, rocksdb) — port the **search space** and pair it with
a deterministic surrogate + fixed seed (RandomSampler → bit-exact), as in
[`examples/`](../../../examples) / `GistExampleParityTests`. **The top gist authors are Optuna maintainers** — their gists are
canonical. Known goldmine: keisuke-umezawa gist `d57c1a2acd4afebe3845ecec22a2b6bc` = ~20 study
archetypes in one script (an oracle-corpus seed). See memory `battle-tested-optuna-usage-sources`.

### 2 — Extract the concrete API surface
```bash
bash $SK/harvest.sh file  kathrinse/TabSurvey models/mlp.py   # raw file from a repo (default branch)
bash $SK/harvest.sh gist  d57c1a2acd4afebe3845ecec22a2b6bc     # gist: desc + file list
bash $SK/harvest.sh gist  d57c1a2acd4afebe3845ecec22a2b6bc tmp.py   # one gist file's content
```
Then `grep -nE 'optuna|suggest_|create_study|Sampler|Pruner|study\.|trial\.'` the file. Hyperparameter
defs are often in a helper (`define_trial_parameters`, `sweep.py`, `objective`), not the matched file —
follow with `harvest.sh` (or `gh search code '<symbol>' repo:owner/name`). Capture: sampler + kwargs
(seed? multivariate/constant_liar/group?), every `suggest_*` (name, bounds, `log`/`step`, choices),
`direction(s)`, pruner, `n_trials`, reports/prune, `enqueue_trial`, user attrs.

### 3 — Gap-check BEFORE writing (don't duplicate)
```bash
grep -rl -i '<pattern>' test/OptunaSharp.Tests.Interop --include=*.cs
```
Already covered thoroughly: every sampler/pruner trajectory, NaN/±inf **intermediate reports**, mixed
categoricals, unicode, single-value & step-adjusted dists, `enqueue_trial`, dynamic/branching spaces,
deep attr trees, extreme float bit patterns. Only port what's **not** there (e.g. inf/-inf objective
**return** values were a gap — now covered). Not byte-reproducible: default (unseeded) TPE — those give
you API-shape coverage only, so pin a `seed`.

### 4 — Observe live Optuna (capture the oracle)
```bash
python - <<'PY'
import optuna; optuna.logging.set_verbosity(optuna.logging.CRITICAL)
s = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42), study_name="x")
s.optimize(<the extracted objective>, n_trials=9)
for t in s.trials: print(t.number, repr(t.value), dict(t.params))
PY
```
Confirm it's deterministic under the seed and note the exact trajectory.

### 5 — Write the parity test (existing idiom)
New file `test/OptunaSharp.Tests.Interop/<Thing>ParityTests.cs`, extend `OptunaInteropTestBase`,
`[TestClass]`, each method `[TestMethod] [TestCategory("RequiresPython")]` (+ `[DataRow(seed)]`), first
line `PythonSession.EnsureOrInconclusive();`. Build the C# study, run the **identical** objective as a
Python string, snapshot both, assert.

```csharp
// Build + run C# side
var cs = Optuna.CreateStudy(storage: new InMemoryStorage(),
    sampler: new RandomSampler(seed), studyName: "x");
cs.Optimize(t => { double v = t.SuggestFloat("x", -10, 10); return Math.Pow(v, 2.0); }, nTrials: 9);
var csState = StudySnapshot.Capture(cs);

// Build + run the SAME thing in live Optuna, dump its full state
PyExec($"study = optuna.create_study(study_name='x', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
       "study.optimize(lambda t: t.suggest_float('x', -10, 10) ** 2, n_trials=9)\n");
var pyState = PyStudyState("study");   // the python var MUST be named `study`

StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);  // expected = python
```

**Bit-exact vs tolerance — pick the assertion:**
- **Bit-exact** (`StudySnapshot.AssertParity`, full-state, `DoubleToInt64Bits`): RandomSampler,
  GridSampler, BruteForceSampler, QMCSampler, CmaEsSampler, NSGA-II/III, PartialFixed.
- **Tolerance trajectory** (branch/state/param-set exact, continuous params ~`1e-9`, non-finite values
  exact): **TPESampler, GPSampler** — they route densities through the FreeBSD `erf` port (~1-3 ULP
  from Python's ucrt `math.erf`). Copy `AssertTrajectoryParity` from `TpeSamplerParityTests.cs` /
  `BattleTestedUsageParityTests.cs`.

**Gotchas that cost byte-parity:**
- Scalar `x**2` in Python = `pow(x,2)` → use `Math.Pow(x, 2.0)`, **never `x*x`** (last-ULP drift). See
  memory `numeric-bitexact-parity-notes`.
- `BeApproximately` is undefined at ±inf — compare non-finite trial values **exactly**.
- RNG discipline: `LazyRandomState`/seeded samplers only, never `System.Random`.
- Pass JSON to Python via `PySet`/scope binding, never string-interpolation into source.
- GIL: any `PyObject` touch (incl. Dispose) inside `using (Py.GIL())` — the base-class helpers already do this.

### 6 — Verify (build fast, then run the live gate)
```bash
# compile only (escape-hatch flags avoid the sibling-NumSharp weaver/pack side-effects)
dotnet build test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release -f net10.0 \
  -p:SkipNDScopeWeave=true -p:GeneratePackageOnBuild=false

# run just the new tests against live Optuna (REQUIRE_ENGINE=1 → discovery failure goes red, not skipped)
PYTHONNET_PYDLL="C:/Users/ELI/.claude/python/python312.dll" OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1 \
  dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release -f net10.0 \
  --filter "FullyQualifiedName~<YourClass>" -p:SkipNDScopeWeave=true -p:GeneratePackageOnBuild=false
```
Green → the real-world usage ports byte-for-byte. **Red is a finding**, not necessarily a bug in the
test: triage whether OptunaSharp genuinely diverges (then fix Core, or document with `[Misaligned]` +
a comment on why). Note the `dotnet` caveat: taskkill isn't available here — if a build wedges on a
locked DLL, pause for the user.

## Definition of done
Test lives under `test/OptunaSharp.Tests.Interop/`, cites its GitHub/gist provenance in the class doc,
runs green via the live gate, and adds coverage that wasn't already there. Consider seeding the future
`OptunaSharp.Tests.Oracle` committed corpus from the same fixture. Commit `feat(interop): …` (or
`test(interop): …`) with an extensive body naming the source repo/gist.

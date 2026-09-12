# OptunaSharp interactive presentations

HTML presentations with a local ASP.NET Core host and live OptunaSharp studies. The first deck explains NSGA-III through two interactive laboratories and an end-to-end algorithm animation.

## NSGA-III algorithm atlas and complete source document

Open the **[Three.js algorithm atlas](http://127.0.0.1:5088/nsga3-atlas/)** for a Desmos-inspired expression rail, a real WebGL 2 objective-space scene, and an IDA-inspired source explorer. The panels synchronize mathematical stages, real OptunaSharp snapshots, and exact C# method ranges. Use **Focus mode** for the presentation view, or **Math / Scene / Code** to expand one instrument.

The **[complete sourcebook](http://127.0.0.1:5088/nsga3-atlas/sourcebook.html)** is a standalone, printable HTML document. Its Markdown source is [NSGA-III-SOURCEBOOK.md](NSGA-III-SOURCEBOOK.md), also downloadable from the browser. It preserves **43 complete files and 8,378 physical lines**, including comments, blank lines, the dedicated NSGA-III sampler and selection strategy, every shared genetic operator, nondominated sorting, normalization support, random sampling, and the listed search-space/distribution helpers. The sourcebook explicitly lists the framework and third-party dependency boundary. This is the full listed algorithm source, not the entire .NET runtime or a standalone replacement for the Core project.

The atlas offers:

- Source-file and function navigation, literal text search, line selection, full-file listings, and stable deep links.
- Function-reference graphs using 57 curated local call relationships, plus the eight-stage teaching flow. Click a function to inspect its complete source.
- Real 3D DTLZ2 geometry, lit trial spheres, an analytic Pareto surface, normalized reference rays, perpendicular projection links, and actual niche choices. Drag to orbit, wheel to zoom, and shift-drag to pan.
- An independent reference-vector laboratory with C# association results. These probe coordinates can vary independently; they need not sum to one.
- Trial inspection by clicking a sphere or selecting a trial from the dropdown, including all three objectives, normalized values, rank, direction, distance, and survivor status.
- Real generation stepping, visual stage stepping, individual niche decisions, play/pause, and recorded generation scrubbing. Source highlights explain completed generations; they do not claim to be a live instruction-level debugger.

The sourcebook generator reads the current working tree and records raw-byte SHA-256 hashes, Git HEAD, exact line counts, member boundaries, and stage links. Regenerate it whenever the included C# changes:

```sh
node docs/presentations/build-sourcebook.mjs
node docs/presentations/build-sourcebook.mjs --check
node docs/presentations/tests/atlas.mjs --http
```

Three.js **0.183.2** and OrbitControls are vendored locally with their MIT license and integrity manifest. Normal viewing requires no CDN or Node build. A WebGL 2 capable browser renders the scene; source and trial inspection remain available if graphics initialization fails. To verify or reproduce the pinned dependency:

```sh
node docs/presentations/vendor-three.mjs --check
# Redownload the exact npm release and verify its published archive integrity:
node docs/presentations/vendor-three.mjs
```

The geometry follows the official [Three.js renderer](https://threejs.org/docs/pages/WebGLRenderer.html) and [OrbitControls APIs](https://threejs.org/docs/pages/OrbitControls.html). Source artifacts live in `wwwroot/nsga3-atlas/`; the renderer, source viewer, request state, and page coordination are separate modules for reuse.

## Run

Install the **.NET 10 SDK**, then run from the repository root:

```powershell
./docs/presentations/run.ps1
```

Open **[NSGA-III — the geometry of better](http://127.0.0.1:5088/nsga3/)** or the **[presentation library](http://127.0.0.1:5088/)**. The host serves both the presentation and API from the same origin. Initial NuGet restore needs internet access; the presentation itself uses local assets. No Node build, Python process, CDN, database, or separately installed Optuna package is required.

The equivalent command, also usable on Linux/macOS with the .NET SDK, is:

```sh
dotnet run --project docs/presentations/OptunaSharp.PresentationHost --configuration Release -- --urls http://127.0.0.1:5088
```

Use `./docs/presentations/run.ps1 -Port 5090` to choose another port. Press **Ctrl+C** in the host terminal to stop it. Serve the HTML through this host: opening `index.html` directly does not provide the live API.

## Explore NSGA-III

1. **Pareto trade-offs:** move two-objective candidates and inspect dominance. Both objectives are minimized. This mathematical playground runs locally in the browser.
2. **Reference directions:** change the simplex divisions and move an objective vector. The C# backend returns the nearest reference ray and its perpendicular projection. More directions give more niches, but do not guarantee every niche receives a survivor.
3. **A complete evolution:** configure a seeded DTLZ2 study and run generations. Play or step through the visual stages of breeding/evaluation, combination, ranking, normalization, association, niche selection, and survival. History and export preserve the actual snapshots returned by the server.

The visual stages replay a completed generation. The backend really runs OptunaSharp; animation speed controls playback speed. A failed API request appears as an error instead of fabricated optimizer results.

The shared controls provide **previous/next section**, **presenter notes**, and **fullscreen** when the browser permits it. **Page Up/Page Down** or **left/right arrows** navigate sections when focus is outside an interactive control; **N** toggles notes. Sliders keep their native keyboard behavior. Reduced-motion preferences disable movement transitions. Fullscreen availability depends on the containing browser.

## What actually runs

The host references `src/OptunaSharp.Core` directly. Each session creates a real in-memory `Study` with `NSGAIIISampler`, simulated binary crossover (`eta=20`, crossover probability `0.9`), polynomial mutation (`eta=20`, configurable per-variable probability), and three minimization objectives. All optimizer state remains in C#.

The benchmark is **DTLZ2**, with 12 decision variables in `[0, 1]`, three objectives, and `k=10` distance variables:

```text
g  = Σ(i=2..11) (x[i] − 0.5)²
f1 = (1 + g) cos(π x[0]/2) cos(π x[1]/2)
f2 = (1 + g) cos(π x[0]/2) sin(π x[1]/2)
f3 = (1 + g) sin(π x[0]/2)
```

Its true Pareto front is the positive octant of the unit sphere. Mean radial distance `mean(||f||₂ − 1)` measures convergence of the current survivors. Occupied reference directions describe their spread. Neither metric alone proves convergence and coverage, and an individual generation need not improve either metric.

Generation `0` contains the initial `N` completed trials. Every **next generation** completes another `N` evaluations. To observe survivor selection without changing random draws or the first offspring's parent ordering, the host begins one next-generation trial and leaves it unevaluated at each snapshot boundary. That trial is completed on the next step. Evaluation counts include only completed trials.

An observer subclass calls the original `SelectParent` exactly once per generation and captures its selected trial IDs and order. Ranking, normalization, association, and niche explanations are reconstructed in the presentation host from these candidates and actual choices, using the same formulas and NumSharp linear algebra as the core implementation. They are presentation diagnostics, not instrumentation inside each private algorithm operation. No sampler source is modified.

Normalization uses the accepted fronts and borderline front together: subtract the ideal point, find ASF extreme points, and fit hyperplane intercepts, with an axis-maximum fallback for degeneracy. Association minimizes perpendicular distance to a ray through the origin. Niche occupancy initially counts only already accepted individuals. An empty niche takes its closest candidate; a populated niche uses the sampler's randomized choice.

## Reuse the infrastructure

```text
docs/presentations/
├── run.ps1                         # Restore/build/run or self-test
├── new-presentation.ps1             # Copy a safe starter into a new deck directory
├── OptunaSharp.PresentationHost/    # Same-origin static host, API, real studies, diagnostics
├── tests/smoke.ps1                  # HTTP integration checks against a running host
└── wwwroot/
    ├── index.html                  # Presentation library
    ├── shared/presentation.css      # Scoped navigation/notes/fullscreen styling
    ├── shared/presentation.js       # Shared accessible presentation behavior
    ├── template/index.html          # Minimal standalone starter
    └── nsga3/                      # Artistic NSGA-III presentation and interactions
```

Create another deck:

```powershell
./docs/presentations/new-presentation.ps1 -Name my-algorithm
```

Edit `wwwroot/my-algorithm/index.html`, then open `http://127.0.0.1:5088/my-algorithm/`. Each deck owns its markup, graphics, styles, and scripts. To opt into the shared shell:

```html
<link rel="stylesheet" href="/shared/presentation.css">
<script defer src="/shared/presentation.js"></script>
<section id="experiment" data-slide data-notes="What the presenter should demonstrate.">
  <!-- Deck content -->
</section>
```

The shell emits `presentation:slidechange` on `document`, with `event.detail.index` and `event.detail.section`. It does not hide other sections or assume a chart library. Add API routes in the host when a future deck needs another real algorithm. Keep experiment state on the server and presentation playback state in the browser.

## Verify

```powershell
# Determinism, objective geometry, sampler parity, and selection invariants:
./docs/presentations/run.ps1 -SelfTest

# With the host running in another terminal:
./docs/presentations/tests/smoke.ps1
```

The HTTP checks cover real run creation and stepping, read-only snapshots, deterministic runs, reference geometry, validation errors, and cleanup. Browser verification should additionally exercise both labs, playback/pause, generation history, restart with changed parameters, and a narrow viewport.

## API

Requests and responses use camel-case JSON. Run configuration:

```json
{"populationSize":36,"divisions":6,"seed":42,"mutationProbability":0.15,"problem":"dtlz2"}
```

| Method | Route | Result |
| --- | --- | --- |
| GET | `/api/health` | Engine and algorithm status |
| POST | `/api/nsga3/runs` | Create a study; return generation 0 |
| GET | `/api/nsga3/runs/{runId}` | Read current snapshot without stepping |
| POST | `/api/nsga3/runs/{runId}/step` | Evaluate one generation; return its snapshot |
| DELETE | `/api/nsga3/runs/{runId}` | Remove the in-memory study |
| POST | `/api/nsga3/reference-lab` | Reference directions and nearest-ray projection |

Reference laboratory request:

```json
{"divisions":6,"point":[0.45,0.35,0.2]}
```

Snapshots contain parents, offspring, candidates, survivors, the current survivors' nondominated subset, reference directions, convergence/occupancy metrics, and selection diagnostics. Trial `id` is the study's trial number; ranks and direction indices are zero-based. The C# records in `Contracts.cs` define the complete schema. Sessions are bounded, serialized per run, expire after inactivity, and disappear when the host stops. The host binds to loopback by default and is intended for local presentations.

## Sources

- Deb & Jain, *An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I* ([DOI](https://doi.org/10.1109/TEVC.2013.2281535), [author-hosted manuscript](https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf)). Algorithms 1–4 define the selection, normalization, association, and niching process.
- [Optuna NSGAIIISampler documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIIISampler.html) and [selection implementation](https://github.com/optuna/optuna/blob/master/optuna/samplers/_nsgaiii/_elite_population_selection_strategy.py).
- [DTLZ2 definition in pymoo](https://pymoo.org/problems/many/dtlz.html#DTLZ2).
- Local implementation: `src/OptunaSharp.Core/Samplers/NSGAIIISampler.cs`, `Samplers/BaseGASampler.cs`, and `Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs`.

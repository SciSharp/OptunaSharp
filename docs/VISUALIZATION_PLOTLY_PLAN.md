# Phase 5.2 — `OptunaSharp.Visualization.Plotly` (raw-C# rendering on Plotly.NET)

> **Status: COMPLETE.** All 12 `optuna.visualization` `plot_*` functions (+ `IsAvailable`) are ported,
> each with the golden-standard Plotly.NET customization overload, and validated **byte-exact vs live
> Optuna** (50 render-parity scenarios, both TFMs). Byte-exact except three documented exclusions: the
> `make_subplots` `domain` geometry (float-op-order boilerplate, N×N contour/rank), the timeline plot's
> wall-clock `base`/`x`, and the GP-derived terminator `y`/`range` (validated in the GP suite); the MO
> parallel-coordinate `draw_order` stays the documented `[Misaligned]`.
>
> **The golden-standard customization overload is itself a parity gate, not just the plain renderers.**
> `PlotlyCustomizationRenderParityTests` (interop) renders every one of the 12 `Plot*(study, customization, …)`
> overloads and diffs it, live, against the same Optuna figure with the equivalent `fig.update_layout(...)`
> applied on the plotly.py side — 24 scenarios across all 12 overloads (incl. a layout-key **override**
> case), both TFMs, so the Plotly.NET `addLayout` merge is proven byte-identical to plotly's own
> `update_layout` on real Optuna figures. `PlotlyChartCustomizationAllPlotsTests` (unit, Python-free) pins
> the same contract for all 12 overloads without Python: config/size applied, every canonical `data` trace
> and `layout` key preserved, DisplayOptions attached, caller keys win on conflict, empty bundle is a no-op.
>
> Renders the already-complete visualization **data layer** (Phase 3.1, in Core's
> `OptunaSharp.Visualization` namespace) into interactive **plotly figures** using
> [Plotly.NET](https://plotly.net/), driven from **raw C#** (bare `Trace`/`Layout` objects + the
> `DynamicObj` escape hatch — not the F# DSL, not the `Chart2D` sugar). One Tier X satellite package,
> pure managed. Port target: `optuna.visualization` (the plotly backend), Optuna @ `f6898da`
> (v5.0.0.dev, vendored at `src/optuna`).

---

## 0. Why Plotly.NET, and why "raw C#"

Optuna's figures **are** plotly.js figures, and Optuna's own visualization tests assert on the figure
**spec** (`figure.data` / `figure.layout`), not on pixels. Plotly.NET models that exact object graph
and serializes to the same plotly JSON, so the C# figure is diff-able against live Optuna's
`figure.to_dict()`. (A raster library such as ScottPlot can only be pixel-compared — the wrong
paradigm for this repo. See the evaluation that preceded this plan.)

**Hands-on verification (done).** Building **bare `Plotly.NET.Trace(type)` and `Layout` objects** and
setting properties with `DynamicObj.SetValue("<exact plotly key>", value)` yields byte-level control of
the figure spec. `GenericChart.toJson` then emits literally `{"data":[…],"layout":{…}}` — every
property Optuna uses round-trips key-for-key, including the two constructs that looked risky:

- Contour's nested `contours:{coloring:"heatmap"}` (from Optuna's `contours_coloring="heatmap"`).
- `make_subplots` layout: multi-axis via `xaxis2`/`yaxis2` + `domain`/`anchor` (exactly how plotly lays
  subplots out).

The complete trace surface across all 15 Optuna renderer files is **only six types** — `Figure`,
`Scatter`, `Contour`, `Bar`, `Scatter3d`, `Layout` — all expressible as bare `Trace`. So "raw C#"
means: **do not** use the `Plotly.NET.CSharp` `Chart2D.Chart.Scatter(...)` helpers (they pre-seed
`line:{}`/`marker:{}` defaults that Optuna does not emit and would pollute the spec diff). Build the
traces ourselves; use Plotly.NET only as the typed figure container + JSON serializer + HTML renderer.

---

## 1. Package identity

| | |
|---|---|
| **PackageId** | `OptunaSharp.Visualization.Plotly` (mirrors `optuna.visualization`; built *on* the Plotly.NET library. If a literal `OptunaSharp.Visualization.Plotly.NET` id is preferred, it's a one-line rename.) |
| **AssemblyName / RootNamespace** | `OptunaSharp.Visualization.Plotly` |
| **TFMs** | `net8.0;net10.0` |
| **Kind** | library, **packable**, Tier X satellite |
| **Deps** | `Plotly.NET` 5.1.0 + `Plotly.NET.CSharp` 0.13.0 (both **pure managed**: DynamicObj, FSharp.Core, Giraffe.ViewEngine, Newtonsoft.Json — zero native), `ProjectReference` → Core |
| **Core friendship** | **Not** a Core friend — the data layer (`OptunaSharp.Visualization.*Info` records + `*.GetInfo*` statics) is already **public**. Uses only Core's public surface. |

csproj is the `OptunaSharp.Interop.BoTorch` template minus pythonnet: shared metadata from
`Directory.Build.props`, README + shared icon packed, `.pdb` in the nupkg, strong-named repo-wide.

```xml
<ItemGroup>
  <PackageReference Include="Plotly.NET" Version="5.1.0" />
  <PackageReference Include="Plotly.NET.CSharp" Version="0.13.0" />
  <ProjectReference Include="..\OptunaSharp.Core\OptunaSharp.Core.csproj" />
</ItemGroup>
```

Add the project to `SciSharp.OptunaSharp.sln` (surgical, single `<Project>` block — a concurrent
session may hold the sln; edit by explicit Edit, never regen). Add `OptunaSharp.Tests` coverage; a
live-interop parity gate goes in `OptunaSharp.Tests.Interop`.

---

## 2. Architecture / file layout

```
src/OptunaSharp.Visualization.Plotly/
├── OptunaSharp.Visualization.Plotly.csproj
├── README.md
├── PlotlyVisualization.cs         # public API: the 13 Plot* statics + IsAvailable (mirrors optuna.visualization.*)
├── Internal/
│   ├── Fig.cs                     # raw-C# figure kit: Trace(type), Layout(), Axis(), ofTraces(), SetLayout(); make_subplots domain math
│   ├── PlotlyColors.cs            # pure port of the plotly.colors helpers we need (see §4)
│   └── ColorScales.cs            # named-colorscale tables: Blues (COLOR_SCALE), Reds, RdYlBu_r
└── Renderers/                     # one file per plot; each maps an Info record -> GenericChart
    ├── OptimizationHistoryPlot.cs
    ├── EdfPlot.cs
    ├── IntermediateValuesPlot.cs
    ├── HypervolumeHistoryPlot.cs
    ├── ParamImportancesPlot.cs
    ├── SlicePlot.cs
    ├── ParetoFrontPlot.cs
    ├── ContourPlot.cs
    ├── RankPlot.cs
    ├── ParallelCoordinatePlot.cs
    ├── TimelinePlot.cs
    └── TerminatorImprovementPlot.cs
```

**Public API** — `public static class PlotlyVisualization` in namespace
`OptunaSharp.Visualization.Plotly`, each method returning `Plotly.NET.GenericChart`, mirroring the
Optuna signatures **name/order/defaults-for-name** (C# has no kw-only args; keep optional params in the
same order):

```csharp
GenericChart PlotOptimizationHistory(Study study, Func<FrozenTrial,double>? target = null,
                                      string targetName = "Objective Value", bool errorBar = false);
GenericChart PlotOptimizationHistory(IReadOnlyList<Study> studies, …);   // Sequence[Study] overload
GenericChart PlotEdf(Study/studies, target=null, targetName="Objective Value");
GenericChart PlotIntermediateValues(Study study);
GenericChart PlotHypervolumeHistory(Study study, IReadOnlyList<double> referencePoint);
GenericChart PlotParamImportances(Study study, IImportanceEvaluator? evaluator = null,
                                  IReadOnlyList<string>? params_ = null, Func<…>? target = null,
                                  string targetName = "Objective Value");
GenericChart PlotSlice(Study study, IReadOnlyList<string>? params_ = null, …);
GenericChart PlotParetoFront(Study study, IReadOnlyList<string>? targetNames = null,
                             bool includeDominatedTrials = true, Func<…>? constraintsFunc = null,
                             Func<FrozenTrial,IReadOnlyList<double>>? targets = null);
GenericChart PlotContour(Study study, IReadOnlyList<string>? params_ = null, …);
GenericChart PlotRank(Study study, IReadOnlyList<string>? params_ = null, …);
GenericChart PlotParallelCoordinate(Study study, IReadOnlyList<string>? params_ = null, …);
GenericChart PlotTimeline(Study study, int? nRecentTrials = null);
GenericChart PlotTerminatorImprovement(Study study, bool plotError = false, …);
static bool IsAvailable => true;   // Plotly.NET is a hard dep; mirrors optuna.visualization.is_available
```

Each `Plot*` calls the Core data-layer entrypoint (below), then hands the info record to the matching
`Renderers/*` mapper. The data layer already emits every warning / `ValueError` (verbatim messages),
so renderers stay pure info→figure.

**Rendering / UX seam** (the `figure.show()` / `write_html()` analog): re-export the Plotly.NET fluent
extensions — `chart.Show()` (opens a browser) and `chart.SaveHtml(path)`; add a thin
`PlotlyVisualization.ToFigureJson(chart)` for the spec (parity + embedding). No new rendering code —
Plotly.NET ships the HTML/plotly.js emitter.

**Customization seam (golden-standard Plotly.NET config) — DONE, the pattern for every `Plot*`.** Each
`Plot*` returns a `GenericChart`, so the Plotly.NET **core** fluent extensions already apply
(`using Plotly.NET;` gives `.WithConfig`/`.WithLayout`/`.WithSize`/`.WithTitle`/…). This package adds
only the gaps: `PlotlyChartExtensions.WithDisplayOptions` (core has no fluent `DisplayOptions` setter)
and `WithCustomization`, plus a `PlotlyChartCustomization` bundle (`Config`/`Layout`/`DisplayOptions`,
layout **merged** via `addLayout` so Optuna's title/axes survive) and a **customization overload** on
each `Plot*` — `Plot*(study, PlotlyChartCustomization customization, …)` — that leaves the canonical
Optuna-faithful signature untouched (the overload takes a required bundle, so no ambiguity). **Every
new renderer gets the same overload for free** (route the canonical result through `WithCustomization`).
**Tested by** `PlotlyCustomizationRenderParityTests` (24 live scenarios, all 12 overloads, both TFMs)
and `PlotlyChartCustomizationAllPlotsTests` (Python-free, all 12 overloads + conflict-override + no-op
bundle). Note: `addLayout` **mutates** the chart's layout in place, so snapshot a canonical figure's JSON
before deriving a customized one from it (the unit suite does this for the GP-backed terminator case).

---

## 3. Data-layer entrypoints each renderer consumes (all already public in Core)

| Plot | Core entrypoint | Returns |
|---|---|---|
| optimization_history | `OptimizationHistory.GetInfoList(studies, target, targetName, errorBar)` | `List<OptimizationHistoryInfo>` |
| edf | `Edf.GetInfo(...)` | `EdfInfo` |
| intermediate_values | `IntermediateValues.GetPlotInfo(study)` | `IntermediatePlotInfo` |
| hypervolume_history | `HypervolumeHistory.GetInfo(study, referencePoint)` | `HypervolumeHistoryInfo` |
| param_importances | `ParamImportances.GetInfos(...)` / `GetInfo(...)` | `List<ImportancesInfo>` |
| slice | `SlicePlot.GetPlotInfo(...)` | `SlicePlotInfo` |
| pareto_front | `ParetoFront.GetInfo(...)` | `ParetoFrontInfo` |
| contour | `Contour.GetInfo(...)` | `ContourInfo` |
| rank | `Rank.GetInfo(...)` | `RankPlotInfo` |
| parallel_coordinate | `ParallelCoordinate.GetInfo(...)` | `ParallelCoordinateInfo` |
| timeline | `Timeline.GetInfo(study, nRecentTrials)` | `TimelineInfo` |
| terminator_improvement | `TerminatorImprovement.GetInfo(...)` | `ImprovementInfo` |

`param_importances` also needs the ported importance evaluators (fANOVA/MDI, already in Core) as the
default `evaluator`; wire the same default Optuna uses.

---

## 4. The one pure-port: `plotly.colors` helpers

The renderers use a handful of `plotly.colors` functions + named scales that Plotly.NET does **not**
need to supply (we set the scale as the exact string/list Optuna passes). Port these as pure C#
(trivial, deterministic — no library dependency). rank's per-point RGB was explicitly deferred to 5.2
by the data layer (`RankPlotInfo` exposes `ColorIdxs` + an `Infeasible` mask, not RGB).

- `sample_colorscale(colormap, values)` → RGB strings — used by `rank`.
- `find_intermediate_color(lo, hi, t)` — used by `_utils._sample_colorscale`.
- `label_rgb` / `unlabel_rgb` — `"rgb(r,g,b)"` ⇄ `(r,g,b)`.
- Named scales as literal tables: `Blues` (the `COLOR_SCALE` used by contour/slice/parallel),
  `Reds`, and `RdYlBu_r` (rank). Copy the exact hex/RGB stop lists from plotly and unit-test the
  sampled values against live plotly to lock them.

Everything else — `colorscale="Blues"`, `"Reds"`, `COLOR_SCALE` as a list — is passed straight through
via `SetValue`.

---

## 5. Per-renderer task list (13)

Difficulty: 🟢 trivial · 🟡 moderate · 🔴 involved. Each references the Optuna plotly source
(`optuna/visualization/_*.py`) to keep the C# diff-able against the Python.

1. **optimization_history** 🟡 — `_optimization_history._get_optimization_history_plot`. `Layout(title,
   xaxis{title:"Trial"}, yaxis{title:targetName})`; markers `Scatter(mode:"markers", error_y=
   {type:"data",array:stds,visible:true}` when `errorBar`); best-value `Scatter(mode:"lines")`;
   feasible/infeasible split; multi-study color cycling. Consumes `List<OptimizationHistoryInfo>`.
2. **edf** 🟢 — `_edf`. Step `Scatter(mode:"lines")` per study over the shared x-grid; `Layout` axes
   `["Objective Value", "Cumulative Probability"]`.
3. **intermediate_values** 🟢 — `_intermediate_values`. One `Scatter(mode:"lines+markers")` per trial;
   axes `["Step","Intermediate Value"]`.
4. **hypervolume_history** 🟢 — `_hypervolume_history`. Single `Scatter(mode:"lines+markers")`; axes
   `["Trial","Hypervolume"]`.
5. **param_importances** 🟡 — `_param_importances`. Horizontal `Bar(orientation:"h")` with
   `error_x` when applicable, `text`/`textposition:"outside"`; one bar-group per objective (MO). Needs
   the evaluator default.
6. **slice** 🟡 — `_slice`. `make_subplots(rows=1, cols=len(params), shared_yaxes=True)`; per-param
   `Scatter(mode:"markers")` with `marker.colorscale=COLOR_SCALE` colored by trial number;
   **`showscale` on the first subplot only**; per-axis `type:"log"` / categorical ticks; the colorbar
   `xpad` offset. Consumes `SlicePlotInfo`.
7. **pareto_front** 🔴 — `_pareto_front`. **2D** → `Scatter`; **3D** → `Scatter3d` +
   `Layout.scene{xaxis_title,yaxis_title,zaxis_title}`; dominated/best/infeasible trace split;
   `targets=`/`axis_order`/`include_dominated_trials`; hovertext. (>3 objectives without `targets`
   raises in the data layer.)
8. **contour** 🔴 — `_contour`. Single pair → one cell; N params → `make_subplots(N×N, shared_x/y)`
   with per-cell `update_xaxes/update_yaxes` (title, range, `type:"log"` via `log10(range)`,
   categorical `categoryorder/categoryarray`). Each cell = `Contour(x=indices,y=indices,z=…,
   colorscale=COLOR_SCALE, connectgaps=true, contours:{coloring:"heatmap"}, hoverinfo:"none",
   line:{smoothing:1.3}, reversescale)` + feasible/infeasible overlay `Scatter`s. **Reconstruct the
   dense `z`**: `z[y][x] = NaN` grid filled from the sparse `ContourInfo…ZValues[(x,y)]` (Optuna:
   `np.full((len_y,len_x),nan); z[xys[:,1],xys[:,0]]=zs`). Emit gaps as JSON `null` (see §6).
9. **rank** 🔴 — `_rank`. `make_subplots` like contour; per-point RGB via the **`PlotlyColors`
   sample_colorscale** over `RankPlotInfo.ColorIdxs` (+ infeasible grey); a **manual colorbar trace**
   `Scatter(mode:"markers", marker={colorscale:"RdYlBu_r", showscale:true, colorbar:{thickness:10,
   tickvals,ticktext}})` added last via `_get_tick_info`.
10. **parallel_coordinate** 🔴 — `_parallel_coordinate`. **NOTE this pinned Optuna rewrote it OFF
    `go.Parcoords` onto plain `Scatter` lines** — a big simplification: normalize each dimension to
    [0,1], draw one `Scatter(mode:"lines")` per trial across integer x-positions, color by target via
    a colored marker line; a hidden colorbar `Scatter`; the per-axis tick labels as **`Layout.
    annotations`** + a custom `xaxis{tickvals,ticktext,range}`. Categorical/log tick synthesis lives in
    the data layer (`ParallelCoordinateInfo` dims carry `tickvals`/`ticktext`). The MO `draw_order` is
    a documented `[Misaligned]`.
11. **timeline** 🟢/🟡 — `_timeline`. Horizontal bars via `Scatter` (or `Bar`) spanning start→complete
    datetimes per trial, colored by state; `hovertext` (already bit-exact JSON from the data layer);
    the running-trial bar; reversed-time.
12. **terminator_improvement** 🟡 — `_terminator_improvement`. Improvement `Scatter(mode:"lines+
    markers")` + optional error band (`fill`, `fillcolor`); `min_n_trials` marker. Consumes
    `ImprovementInfo` (the data half is Phase 3.3 `TerminatorImprovement.GetInfo`).

---

## 6. Known parity details (verify during implementation, not blockers)

- **Structural, not byte-identical whole-figure JSON.** Plotly.NET and plotly.py inject different
  layout defaults (Plotly.NET's `toFigureJson` adds a full `template` + `width/height/frames`; use the
  lean `toJson` = `{data,layout,config}` and strip/ignore the root `config:{}` — Optuna's `to_dict()`
  is `{data,layout}`). The gate compares **`figure.data` traces field-by-field + the meaningful
  `layout` fields** (axis titles/types/ranges, annotations, showlegend, scene titles), which is exactly
  Optuna's own test methodology. Property *ordering* is irrelevant to a structural diff.
- **NaN gaps in contour/rank `z`.** plotly.py serializes `np.nan` → JSON `null`. Confirm Plotly.NET's
  serializer for `double.NaN`; if it throws or emits `NaN`, build the grid as `object?[][]` with
  `null` for gaps so the wire matches.
- **Avoid Plotly.NET default injection.** Bare `Trace(type)` only — never the `Chart2D.Chart.*`
  helpers (they add `line:{}`/`marker:{}`). Verified: bare traces serialize clean.
- **Log axes:** Optuna sets `type:"log"` and (contour) a `log10`-transformed `range`. Mirror the
  `math.log10` step in the renderer, not the data layer.
- **`is_available`** is trivially `true` (hard dep) — mirror Optuna's symbol for API compatibility.

---

## 7. Testing

- **Unit (`OptunaSharp.Tests/Visualization/Plotly/`)** — deterministic info→figure: build a small
  study, render, assert on the `GenericChart`'s serialized `data`/`layout` (trace count, modes, x/y
  arrays, marker colors, axis types/titles, annotations). Reuses fixtures already used by the data-
  layer tests.
- **Live-interop parity (`OptunaSharp.Tests.Interop/…PlotParityTests.cs`)** — the real gate. For each
  `plot_*`: build the same study in C# and Python (the existing `OptunaInteropTestBase` idiom), call
  live `optuna.visualization.plot_x(study).to_dict()`, JSON-normalize both, and structurally compare
  `data` + meaningful `layout` (a `PlotlyFigureComparer` helper: recurse dicts/lists, compare numbers
  with the repo's bit-exact-or-erf tolerance, ignore Plotly.NET-only keys like root `config`). Category
  `[RequiresPython]`; self-skips without Python, hard-fails in CI (`OPTUNASHARP_PYTHONNET_REQUIRE_
  ENGINE=1`).
- The numeric arrays are already bit-exact (Phase 3.1); 5.2 verifies the **assembly** into traces.

---

## 8. Sequencing & effort

1. Scaffold package + csproj + sln + README + `Fig.cs` + `PlotlyColors.cs`/`ColorScales.cs` + the
   `PlotlyFigureComparer` test helper. Prove the loop end-to-end on **optimization_history** (unit +
   interop).
2. 🟢 batch: edf, intermediate_values, hypervolume_history, timeline.
3. 🟡 batch: param_importances, slice, terminator_improvement.
4. 🔴 batch: contour, rank, parallel_coordinate, pareto_front (the subplot/colorbar/3D machinery — do
   contour first; rank reuses its subplot grid).
5. Wire `PlotlyVisualization.Show/SaveHtml/ToFigureJson`; DocFX `visualization.md` update; add an
   `examples/` showcase (renders a study to HTML).

**Estimate:** ~1,500 C# SLOC (matches PORTING_PLAN §5.2), ~1–2 weeks. Low risk — the hard numeric work
is done; this is mechanical mapping + one small color pure-port, all against a proven library seam.

**PORTING_PLAN alignment:** this fills §5.2 (`Visualization rendering → .Visualization.Plotly`). Update
the §5.2 row + Tier-X package-map row to point here once the package lands (leave the shared
`PORTING_PLAN.md` edit to a moment no concurrent session holds it).

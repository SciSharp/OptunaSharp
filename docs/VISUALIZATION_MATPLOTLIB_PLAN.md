# Phase 5.2b — `OptunaSharp.Visualization.Matplotlib` (drive real Optuna matplotlib via pythonnet)

> **Status: PLAN.** Ships Optuna's **second** visualization backend —
> `optuna.visualization.matplotlib` — to OptunaSharp by driving the **real** matplotlib backend of the
> **real** Optuna **in-process via pythonnet**, exactly like `OptunaSharp.Interop.BoTorch` drives real
> BoTorch (committed Phase 6.5-A, `803242d`) and `OptunaSharp.Interop.pythonnet` drives real OptunaHub.
> One Tier X satellite; foreign-runtime (embedded CPython + matplotlib). Port target:
> `optuna/visualization/matplotlib/*.py`, Optuna @ `01ddd17` (the v5.0.0 release; written against the
> pre-release `f6898da` — `optuna/visualization` did not change between the two).

---

## 0. The strategic call: drive the genuine article, don't re-port matplotlib

Two facts decide the whole design:

1. **Optuna's matplotlib backend is EXPERIMENTAL** — every function carries
   `@experimental_func("2.2.0")` — and returns a matplotlib `Axes` (`Axes[]` for the contour/rank
   grids, `Axes3D` for 3-objective pareto). A matplotlib `Axes` has **no serializable spec**, so —
   unlike the plotly backend — it is **not structural-parity-checkable**. Pixel comparison is the only
   option, and it is font/AA/platform-fragile (against this repo's byte-exact ethos).

2. It is a **redundant convenience** sitting on the *same* pure-numpy data layer as the plotly backend
   (each matplotlib `plot_*` imports the identical `_get_*_info` builder).

Given (1)+(2), the right move mirrors the BoTorch decision (Eli: *don't re-port when a foreign runtime
is the actual value*): **reconstruct the study Python-side and call the real
`optuna.visualization.matplotlib.plot_*`**, handing back the matplotlib figure. This yields output that
is **definitionally identical to Optuna's** (it *is* Optuna) for a few hundred lines of bridge glue —
versus re-porting 15 files of `ax.plot/scatter/contourf/colorbar/legend` calls through pythonnet for
zero parity payoff. Call this **Strategy M1** (chosen). Strategy **M2** (marshal the *native* Core info
records into a pythonnet matplotlib-drawing port, no Optuna-Python dependency) is documented in §6 and
**deferred** — more code, no fidelity gain, and it still needs matplotlib anyway.

**Consequence:** unlike the plotly package, this satellite does **not** consume Core's native
visualization data layer at runtime (real Optuna recomputes the info from the reconstructed study).
Core is referenced only for the `Study`/`FrozenTrial`/distribution types being marshaled.

---

## 1. Package identity

| | |
|---|---|
| **PackageId / AssemblyName / RootNamespace** | `OptunaSharp.Visualization.Matplotlib` |
| **TFMs** | `net8.0;net10.0` |
| **Kind** | library, **packable**, Tier X satellite (foreign runtime) |
| **Deps** | `ProjectReference` → `OptunaSharp.Interop.pythonnet` (brings `pythonnet [3.2.1, 4.0.0)` transitively + all engine/GIL/discovery/marshaling plumbing) and → Core |
| **Runtime prereq** | a discoverable CPython with `pip install optuna matplotlib` (self-describes + throws a helpful message if absent — the `PythonRuntime` pattern) |
| **Core friendship** | not required (uses Core's public surface + PyConvert) |

csproj = the `OptunaSharp.Interop.BoTorch` template verbatim (it is the closest sibling: a pythonnet
satellite that drives a real Python lib on top of `Interop.pythonnet` + Core). Add to the sln
surgically; add interop-smoke tests.

---

## 2. Architecture / file layout

```
src/OptunaSharp.Visualization.Matplotlib/
├── OptunaSharp.Visualization.Matplotlib.csproj
├── README.md
├── MatplotlibVisualization.cs   # public API: the 13 Plot* statics + IsAvailable (mirrors optuna.visualization.matplotlib.*)
├── MatplotlibFigure.cs          # GIL-safe opaque handle around the returned Axes/Figure PyObject
└── (reuses PyStudyBridge from Interop.pythonnet — see §3)
```

**Public API** — `public static class MatplotlibVisualization` in namespace
`OptunaSharp.Visualization.Matplotlib`, one method per Optuna function, same names/order/defaults as the
plotly package (§ Plotly plan), each returning a **`MatplotlibFigure`** handle instead of a
`GenericChart`. Plus `static bool IsAvailable` mirroring
`optuna.visualization.matplotlib.is_available` (checks matplotlib importability, not a hard `true` —
matplotlib is a *runtime* pip dep, and Optuna's symbol reports real availability).

Each method body is the same three GIL-scoped steps:

```csharp
public static MatplotlibFigure PlotOptimizationHistory(
    Study study, Func<FrozenTrial,double>? target = null,
    string targetName = "Objective Value", bool errorBar = false)
    => PythonRuntime.Run(() =>
    {
        using var pystudy = PyStudyBridge.ToPythonStudy(study);          // 1. reconstruct
        using var mpl = PythonRuntime.Optuna.GetAttr("visualization").GetAttr("matplotlib");
        using var fn  = mpl.GetAttr("plot_optimization_history");
        using var kwargs = new PyDict {
            ["target_name"] = targetName.ToPython(),
            ["error_bar"]   = errorBar.ToPython(),
            // target: pass a Python callable wrapping the C# Func, or omit when null (see §4)
        };
        PyObject axes = fn.Invoke(new[] { pystudy }, kwargs);            // 2. call REAL Optuna
        return MatplotlibFigure.Adopt(axes);                            // 3. wrap (keeps the PyObject alive)
    });
```

---

## 3. The load-bearing primitive: `PyStudyBridge.ToPythonStudy(Study)`

The reconstruction already exists **privately** in `OptunaHubSampler` (`BuildStudy` + `CreateFrozenTrial`
+ `PyTrialState`): it does `optuna.create_study(directions=[…])` then, per finished C# `FrozenTrial`,
`optuna.trial.create_trial(state, values, params, distributions)` + `study.add_trial(...)`.

**Task:** promote that into a shared, public helper in `OptunaSharp.Interop.pythonnet`
(`PyStudyBridge.ToPythonStudy(Study, includeRunning: bool = false)`), reused by both the OptunaHub
shadow study and this package (refactor `OptunaHubSampler.BuildStudy` to call it — no behavior change).

**Extend it** — the shadow-study version only carried `state/values/params/distributions` (all a
sampler needs). Several plots need more; `optuna.trial.create_trial` accepts all of it:

- `intermediate_values` — **required** by `plot_intermediate_values`.
- `datetime_start` / `datetime_complete` — **required** by `plot_timeline` (and improves others).
- `user_attrs` / `system_attrs` — `system_attrs["constraints"]` drives feasible/infeasible coloring in
  optimization_history / contour / rank / pareto_front; `user_attrs` for hovertext.
- Study-level `user_attrs` / `system_attrs` and, for MO, the full `directions`.

Marshaling of values/params/distributions reuses `PyConvert.ToPy`. Datetimes → Python `datetime`
(naive-UTC, matching how Core stores them). Unit-test the round-trip by dumping the reconstructed
Python study with the interop harness's `_oss_dump_state` and asserting it equals the source C# study
(the harness already has `StudySnapshot` for exactly this).

> Reconstruction is the single risk surface. It is well-trodden — the OptunaHub bridge and every interop
> test rebuild studies this way — but the *extended* fields (datetimes, intermediate values, constraints)
> are new; §5 pins them with a round-trip test before any plot is wired.

---

## 4. `MatplotlibFigure` — a GIL-safe opaque handle

`plot_*` returns a live `matplotlib.axes.Axes` (or an ndarray of them / `Axes3D`). **GIL discipline**
(from `PythonRuntime`/the interop base): a `PyObject` — including its `Dispose` — must be touched only
under `Py.GIL()`, and must never be handed out of a `Run` body. So wrap it:

```csharp
public sealed class MatplotlibFigure : IDisposable
{
    private PyObject? _axesOrArray;                 // the plot_* return
    internal static MatplotlibFigure Adopt(PyObject axes) { … }

    public void SaveFig(string path, int dpi = 100, string? bbox = "tight")   // fig.savefig(path, …)
        => PythonRuntime.Run(() => { using var fig = ResolveFigure(); /* fig.savefig(...) */ });

    public byte[] ToPng(int dpi = 100)              // savefig to a BytesIO -> byte[]
        => PythonRuntime.Run(() => { … });

    public void Show()                              // matplotlib.pyplot.show()
        => PythonRuntime.Run(() => { using var plt = Py.Import("matplotlib.pyplot"); plt.InvokeMethod("show"); });

    public T Use<T>(Func<PyObject, T> body)         // escape hatch: run arbitrary mpl on the Axes under the GIL
        => PythonRuntime.Run(() => body(_axesOrArray!));

    public void Dispose() => PythonRuntime.Run(() => { _axesOrArray?.Dispose(); _axesOrArray = null; });
}
```

- `ResolveFigure()` handles the three return shapes: a single `Axes` → `axes.get_figure()`; an
  `np.ndarray`/list of Axes (contour/rank grids) → `axes.flatten()[0].get_figure()`; `Axes3D` → its
  `.figure`. All share one `Figure`, so `savefig` on it captures the whole plot.
- **Backend:** default to the non-interactive **`Agg`** backend for `SaveFig`/`ToPng` (headless/CI-safe:
  `matplotlib.use("Agg")` before the first pyplot import); `Show()` is best-effort and needs an
  interactive backend on the host. Set the backend once in `PyStudyBridge`/a small init, opt-out via env.
- `ToPng` returning `byte[]` is the ergonomic .NET path (embed in a report, write anywhere); `SaveFig`
  is the direct `fig.savefig` analog.

---

## 5. The 13 wrappers + `IsAvailable`

Thin — each is the 3-step body of §2 with the right function name + kwargs. Signatures mirror the
matplotlib backend's public functions (identical to the plotly ones **except** `plot_pareto_front`'s MO
handling and that all return `MatplotlibFigure`). kwargs pass straight through as marshaled
Python objects; `params`/`target_names` as `PyList`, booleans/strings/ints via `ToPython()`.

- `target` / `constraints_func` / `targets` callables: wrap the C# `Func<FrozenTrial,…>` as a Python
  callable that receives Optuna's `FrozenTrial` and calls back into C# — **or**, simpler and
  sufficient for v1, support only the common no-callback path and throw
  `NotSupportedException("pass target= via the native OptunaSharp.Visualization.Plotly backend")` when a
  delegate is supplied (callbacks crossing the boundary per-trial are slow and rarely used with the
  matplotlib backend). Decide during implementation; the reconstructed study carries everything the
  no-callback path needs.
- `plot_param_importances`: Optuna's default evaluator (fANOVA) runs Python-side on the reconstructed
  study; a caller-supplied `evaluator` maps to the corresponding Optuna evaluator name (or the
  throw-on-custom rule above).
- `IsAvailable` → call `optuna.visualization.matplotlib.is_available` and return it (reflects whether
  matplotlib is installed in the host interpreter).

| # | wrapper | Optuna fn | notable kwargs |
|---|---|---|---|
| 1 | PlotOptimizationHistory | plot_optimization_history | target, target_name, error_bar |
| 2 | PlotEdf | plot_edf | target, target_name |
| 3 | PlotIntermediateValues | plot_intermediate_values | — (needs intermediate_values in reconstruction) |
| 4 | PlotHypervolumeHistory | plot_hypervolume_history | reference_point |
| 5 | PlotParamImportances | plot_param_importances | evaluator, params, target, target_name |
| 6 | PlotSlice | plot_slice | params, target, target_name |
| 7 | PlotParetoFront | plot_pareto_front | target_names, include_dominated_trials, constraints_func, targets |
| 8 | PlotContour | plot_contour | params, target, target_name (returns `Axes[]` grid) |
| 9 | PlotRank | plot_rank | params, target, target_name (grid) |
| 10 | PlotParallelCoordinate | plot_parallel_coordinate | params, target, target_name |
| 11 | PlotTimeline | plot_timeline | n_recent_trials (needs datetimes) |
| 12 | PlotTerminatorImprovement | plot_terminator_improvement | plot_error, improvement_evaluator, error_evaluator, min_n_trials |

---

## 6. Rejected/deferred alternative — M2 (native info → pythonnet matplotlib drawing)

Port the 15 matplotlib files' *drawing* to C#: build the native Core `*Info` record, marshal it into
Python, and issue `ax.plot/scatter/contourf/bar/colorbar/legend/set_xscale(...)` via pythonnet.
Pros: reuses the bit-exact native data layer; no dependency on Optuna's Python viz code (only
matplotlib). Cons: **much** more pythonnet code, all of it untestable beyond a smoke check (same
no-spec-parity problem), and it still needs a matplotlib runtime — so it buys nothing M1 doesn't, at
several times the cost. **Deferred**; revisit only if a "no optuna Python package, matplotlib only"
constraint ever appears (it hasn't for BoTorch/OptunaHub).

---

## 7. Testing

Parity is **not** byte/spec (impossible for `Axes`). Gate on *smoke + structural-on-the-Axes*, all
`[TestCategory("RequiresPython")]` (self-skip without Python; CI hard-fails via
`OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1`, and the interop lane pip-installs `matplotlib`):

- **Reconstruction round-trip** (the real risk): `PyStudyBridge.ToPythonStudy` → `_oss_dump_state` →
  `StudySnapshot` equals the source C# study, across SO/MO, constraints, pruned trials, intermediate
  values, datetimes, attrs. Bit-exact where deterministic.
- **Per-plot smoke:** each `Plot*` returns without raising; the handle resolves a `Figure`; `ToPng`
  yields a non-trivial PNG (magic bytes + size); the Axes structure matches Optuna's own for that plot
  (e.g. `len(fig.axes)` == expected subplot count; `ax.get_xlabel()`/`get_title()` equal the strings
  Optuna sets — compare against calling the same Optuna fn on the same reconstructed study *inside the
  test*, which is a true equivalence check since both sides are the identical Python call).
- **`IsAvailable`** true when matplotlib present.

Because M1 *is* Optuna, the per-plot "does OptunaSharp match Optuna" question reduces to "did we
reconstruct the study faithfully and pass kwargs through" — which the round-trip + kwarg tests cover
directly.

---

## 8. Sequencing & effort

1. Promote + extend `PyStudyBridge.ToPythonStudy` in `Interop.pythonnet` (refactor `OptunaHubSampler`
   onto it — no behavior change) + the round-trip test. **This is the bulk of the real work.**
2. `MatplotlibFigure` handle (Agg backend, ResolveFigure, SaveFig/ToPng/Show/Use/Dispose).
3. Scaffold the package + csproj + sln + README; wire `PlotOptimizationHistory` end-to-end (smoke +
   Axes-equivalence test) to prove the loop.
4. Fill the remaining 12 wrappers (near-mechanical); handle the reconstruction fields each needs
   (intermediate_values → #3, datetimes → #11, constraints → feasibility-aware plots).
5. Decide the `target=`/callback policy (pass-through vs throw); `IsAvailable`; DocFX
   `visualization.md` note that this backend is experimental + needs `pip install matplotlib`; an
   `examples/` showcase saving a PNG.

**Estimate:** ~400–600 C# SLOC (most of it the extended reconstruction + handle; the 13 wrappers are
tiny). ~3–5 days. Risk concentrated entirely in the study reconstruction (§3), pinned by the
round-trip test before any plot is trusted.

**Relationship to the plotly package:** independent and complementary. The native
`OptunaSharp.Visualization.Plotly` is the **primary, offline, parity-checkable** renderer; this
matplotlib bridge is the **optional, experimental** second backend for users who want genuine Optuna
matplotlib figures (papers, existing pipelines) and already have a Python/matplotlib runtime — the same
audience split Optuna itself has.

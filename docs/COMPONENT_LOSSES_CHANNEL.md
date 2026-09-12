# The `component_losses` observed-components channel

> **STATUS: DONE ✅ (Plan 02).** The observed-components channel is a first-class, documented API:
> `trial.SetComponentLosses(...)` (writer, on `BaseTrial`) and `FrozenTrial.TryGetComponentLosses(...)`
> (reader), both backed by the `component_losses` user attribute and centralized in the
> `OptunaSharp.ComponentLosses` helper. `JanusSampler`'s component mode reads it through this single
> reader (the previously inlined `JanusOptimizer.ReadComponents` is gone), and the multi-objective JANUS
> metric (Plan 05, Phase 3) will consume the **same** channel. Pure convention + reader/writer +
> validation — **no new dependency, no Core algorithm change** beyond centralizing the reader.
> Implements [`docs/plans/02-component-losses-channel.md`](plans/02-component-losses-channel.md).

## What it is

Some black-box objectives decompose into a vector of per-trial **residual components**
`r = (r_1, …, r_m)`, where the scalar objective is their mean:

```
f(x) = (1/m) · Σ_i r_i(x)
```

For an objective like this, the residual vector carries far more geometric signal than the scalar
alone — it is where a Jacobian estimate is **sharpest**. JANUS (arXiv:2608.22862) makes this its
primary, recommended mode: given the components it fits the local residual Jacobian `J` directly and
drives its Gauss–Newton exploitation + CMIX exploration from `JᵀJ`; without them it falls back to a
scalar linear/quadratic residual-coordinate fit.

The **observed-components channel** is the convention by which an objective hands that vector to the
sampler: it is stored on the trial as the `component_losses` user attribute, and read back through one
shared, validated reader so that every consumer sees identical semantics.

## API

| Member | Where | Purpose |
|--------|-------|---------|
| `void BaseTrial.SetComponentLosses(ReadOnlySpan<double> components)` | any trial | Validate and store the residual vector on the `component_losses` user attr. |
| `bool FrozenTrial.TryGetComponentLosses(out double[] components)` | completed trial | Read the residual vector back (list-or-array tolerant), or `false` when absent/invalid. |
| `ComponentLosses.Key` (`"component_losses"`) | static | The canonical wire key. |
| `ComponentLosses.LegacyKey` (`"rae"`) | static | Recognized **read** alias. |
| `bool ComponentLosses.TryRead(IReadOnlyDictionary<string,object?>, out double[])` | static | The reader over any raw user-attr dictionary (what the metric consumers call). |

`SetComponentLosses` is simply `SetUserAttr("component_losses", …)` with the contract enforced, so the
value round-trips through the exact same user-attr path every storage (in-memory, RDB, journal, gRPC)
already serializes — the channel is **storage-agnostic**.

## Contract

- **Decomposition semantics.** Components are the per-trial terms of a *mean* decomposition,
  `f ≈ mean_i r_i`. Consumers exploit this — e.g. JANUS treats the best window point by the mean of its
  components.
- **Nonnegative magnitudes.** The terms are treated as nonnegative residual magnitudes (JANUS uses
  `√max(r, 0)`). A negative entry is **permitted** but read by the algorithm at its clamp of zero, so
  the writer does not reject negatives.
- **Consistent length.** All trials that carry components should use a **consistent length** `m` across
  the study. The channel itself is per-trial (it never enforces cross-trial consistency); the *consumer*
  decides what to do when trials disagree.
- **Non-finite rejected.** A vector containing `NaN` or `±∞` is rejected. The **writer** fails fast with
  `ValueError` (so a stored vector always reads back), and the **reader** treats a stored non-finite /
  empty / non-numeric / `null`-bearing value as **absent** (`TryGet…` returns `false`) so a consumer
  degrades gracefully rather than propagating a `NaN`.
- **Mixed-window fall-back.** When only *some* trials in a consumer's window carry valid components, the
  consumer falls back to its scalar mode. JANUS does exactly this: `hasComponents` is true only when
  **all** window trials carry components of the same width `m`; otherwise it uses the scalar
  residual-coordinate metric (`pseudoMode`).

## Reference portability

The underlying key is kept **verbatim** as `component_losses` (with `rae` accepted as a read alias),
matching the Python reference `janus_optuna`
([hongyuanyu/JANUS](https://github.com/hongyuanyu/JANUS), Apache-2.0) and the OptunaHub `janus_moo`
sampler, whose harvest reads `t.user_attrs.get("component_losses") or t.user_attrs.get("rae")`. An
objective written against OptunaSharp's typed channel therefore runs **unchanged** against the Python
reference, and vice-versa — the reader reproduces the reference's Python-`or` fall-through (a canonical
value that is empty/invalid falls through to `rae`).

> **Principled divergence:** the reference does not reject non-finite components at read time (it drops
> non-finite rows later, inside the fit). OptunaSharp rejects them up front (writer) and treats a stored
> non-finite vector as absent (reader). This is strictly safer — JANUS's downstream `AllFinite` guards
> would discard such a fit anyway — and JANUS is convergence-validated (no bit-trajectory to preserve).

## Worked example — JANUS component mode

Expose the residual vector on each trial and JANUS fits its Jacobian directly from it:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 0, nStartupTrials: 16));

study.Optimize(t =>
{
    // f(x) = mean_i (x_i - 0.5)^2 — a component-decomposed sphere.
    var r = new double[10];
    double f = 0;
    for (int i = 0; i < 10; i++)
    {
        double x = t.SuggestFloat($"x{i}", -5, 5);
        r[i] = (x - 0.5) * (x - 0.5);   // the i-th residual component (nonnegative)
        f += r[i];
    }
    t.SetComponentLosses(r);            // hand JANUS the residual vector
    return f / r.Length;
}, nTrials: 400);

// A consumer / analysis reading the components back off a completed trial:
foreach (var trial in study.Trials)
    if (trial.TryGetComponentLosses(out double[] components))
        // components is a fresh double[] of length 10
        Analyze(trial.Number, components);
```

Reading from a raw user-attr dictionary (what the multi-objective metric does over its window):

```csharp
if (ComponentLosses.TryRead(frozenTrial.UserAttrs, out double[] r))
    FitComponentMetric(r);   // else: scalar fall-back
```

## Consumers

- **`JanusSampler` (single-objective, done).** `JanusOptimizer.Harvest` reads the channel per completed
  trial; component mode requires all window trials to carry components of the same width, else scalar
  mode (`JanusMetric`).
- **Multi-objective JANUS metric (Plan 05, Phase 3).** `JanusMooCmixSampler` will read the **same**
  `component_losses` key over its window for `_fit_component_metric`, with the `diag_quadratic` scalar
  fallback — through this identical reader, so the two never drift.

## Validation

- **Unit** (`test/OptunaSharp.Tests/ComponentLossesTests.cs`): writer→reader round-trip on a
  `FrozenTrial` and through a live in-memory study; round-trip through the **JSON wire form** RDB/journal
  storages serialize (`double[]` → `List<object?>` of `long`/`double`); list-vs-array tolerance
  (`double[]`, `List<object?>`, integer forms, `int[]`, `float[]`, mixed); non-finite/empty rejection on
  both sides; the `rae` alias with Python-`or` fall-through; per-trial reads across a mixed window; and
  two JANUS integration checks (typed-writer convergence + mixed-presence scalar fallback).
- **Regression:** the existing `JanusSamplerTests` component-mode gates keep passing after the reader was
  centralized (they still write the raw `component_losses` user-attr, proving the wire convention is
  unchanged).

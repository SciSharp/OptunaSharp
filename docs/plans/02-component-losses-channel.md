# Plan 02 — `component_losses` observed-components channel

> **Status: DONE ✅.** Depends on: nothing (Core only). Tier N. Value ★ / effort low.
> Foundational interface — JANUS's component mode (already reads it) and the MOO metric both consume it.
> Implemented: `OptunaSharp.ComponentLosses` (shared reader/writer + validation),
> `BaseTrial.SetComponentLosses` / `FrozenTrial.TryGetComponentLosses`; `JanusOptimizer` switched to the
> shared reader (inline `ReadComponents` removed). Docs: [`docs/COMPONENT_LOSSES_CHANNEL.md`](../COMPONENT_LOSSES_CHANNEL.md).
> Tests: `test/OptunaSharp.Tests/ComponentLossesTests.cs` (11) + JANUS regression, both TFMs.

## Goal

Formalize the **observed-components channel**: the convention by which an objective exposes a
per-trial residual/component vector `r = (r_1,…,r_m)` for an objective that decomposes as
`f(x) = (1/m) Σ_i r_i(x)`. JANUS Phase 1 already reads a magic `component_losses` user-attr; this plan
turns that into a small, documented, first-class API without breaking the wire convention.

## Motivation

- Component residuals are where JANUS's Jacobian signal is *sharpest* (the paper's primary/recommended
  mode; the UAV and ISP deployments use it). Today it is an undocumented string key
  (`trial.SetUserAttr("component_losses", double[])`) discovered only by reading the sampler.
- The MOO JANUS variant (Plan 05, Phase 3) reads the **same** key. A single typed helper avoids two
  ad-hoc readers drifting.
- Keeping the underlying user-attr key exactly `component_losses` preserves **objective portability**:
  the same objective runs against the Python reference `janus_optuna` unchanged.

## Scope / deliverables

- A thin helper API on the trial surface, e.g. `trial.SetComponentLosses(ReadOnlySpan<double>)` and a
  reader `FrozenTrial.TryGetComponentLosses(out double[])`, both backed by the `component_losses`
  user-attr (last-wins, list-or-array tolerant, non-finite rejected) — i.e. the exact reader currently
  inlined in `JanusOptimizer.ReadComponents`, promoted to one shared place.
- The documented contract: nonnegative per-component terms, consistent length across trials, semantics
  `f ≈ mean_i r_i`; behavior when partially present (the current JANUS choice: fall back to scalar
  mode when not *all* window trials carry components — see the audit's "mixed-components edge case").
- A short doc page under `docs/` (or a section in the JANUS guide) with a worked example.

## Design notes

- Pure convention + a reader/writer + validation; **no** new dependency, **no** Core algorithm change
  beyond centralizing the existing reader. `JanusOptimizer` switches to the shared reader (single
  source of truth).
- Keep it storage-agnostic: the value round-trips through the same user-attr path RDB/journal storages
  already serialize.

## Validation

- Unit tests: set/get round-trip (in-memory + RDB), list-vs-array tolerance, non-finite rejection,
  length-consistency handling.
- Regression: JANUS Phase 1 component-mode tests keep passing after the reader is centralized.

## Out of scope

- A typed multi-fidelity / partial-observation channel (future); this is only the flat residual vector.
- Changing the wire key away from `component_losses` (would break reference portability).

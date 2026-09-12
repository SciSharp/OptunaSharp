# Plan 03 — JANUS infill-wrapper reusable infrastructure

> **Status: DONE.** Depends on: JANUS Phase 1 (done). Tier N. Value ★★ / effort medium.
> Enabling infra for Plan 05 (JANUS Phase 2/3): lets DE and MOO hosts reuse the infill layer.

## Goal

Generalize Phase 1's CMA-ES–specific `JanusOptimizer` into a **host-agnostic infill abstraction**: a
population host (which knows how to ask a generation and tell results) plus an infill provider (which
contributes a few Jacobian-aligned candidates and, optionally, reshapes the proposal distribution).
The point is that JANUS-DE (Plan 05 Phase 2) and JANUS-MOO (Plan 05 Phase 3) plug into the **same**
infill machinery instead of each re-implementing the window/harvest/generation loop.

## Motivation

- Phase 1 entangles three concerns in `JanusOptimizer`: (a) the JANUS math (window → `JᵀJ` → GN elite
  + CMIX candidates — already factored via `JanusMetric`/`FitComponentJacobian`), (b) the CMA-ES host
  driving (ask/tell, coherent-generation batching, resample), and (c) the glue. Only (a) is
  host-independent today.
- The paper is explicit that JANUS is *host-agnostic infill*. To honor that in OptunaSharp we need the
  driving concern (b) behind an interface so a DE or MOO host swaps in.
- Bonus: the abstraction is reusable for **future** infill strategies (not just JANUS) around any
  population host.

## Scope / deliverables

- An internal `IPopulationHost` abstraction capturing the minimum JANUS needs: sample a native
  generation, tell a coherent generation, and expose (where available) the proposal statistics CMIX
  uses — mean, scale `σ`, covariance `C`, and a whitening for bounded injection. Hosts that lack a
  covariance (DE) expose box/step-length bounds instead (the paper's "analogous box and step-length
  constraints").
- A host-agnostic `JanusInfill` driver: harvest window → (fuse cadence) fit `J` → GN bounded-inject +
  CMIX candidates into reserved slots → hand the generation to the host. This is Phase 1's
  `Generate`/`Ask`/`UpdateEs` with the CMA-ES specifics moved behind `IPopulationHost`.
- Refactor `JanusSampler` (CMA-ES) onto the abstraction with **no behavioral change** (the Phase-1
  tests remain the contract).

## Design notes

- Keep the coherent-generation batching + resample-based native sampling (Phase 1's documented
  correctness fixes) in the CMA-ES host implementation; the abstraction must not reintroduce the
  reference's startup-remainder misalignment.
- Bounded injection stays square-root-invariant (Mahalanobis via `C⁻¹`); for hosts without an explicit
  `C`, injection degrades to the host's box/step constraint (DE).
- CMIX needs `mean`/`σ`/`C`; make these optional on the host — a host that can't supply a covariance
  simply doesn't offer CMIX candidates (GN-only infill), which is a valid JANUS-GN configuration.

## Validation

- **Refactor-parity**: all Phase-1 `JanusSamplerTests` and `JanusMetricParityTests` pass unchanged
  (byte-identical elite/candidate stream on a fixed seed) — the abstraction is a pure extraction.
- New unit test: a trivial mock host proves the driver's slot reservation / coherent batching in
  isolation from CMA-ES.

## Out of scope

- Building the DE or MOO hosts themselves (Plans 04, 05). This plan only carves the seam they plug into.

## Outcome (done)

Delivered as a pure, behavior-preserving extraction:

- **`Samplers/Janus/IPopulationHost.cs`** — the seam: `PopulationSize`, `Mean`, `Sigma`, `Ask()`,
  `Tell(generation)`, plus the *optional* proposal-covariance surface `HasProposalCovariance`,
  `ProposalCovarianceTrace`, and `WhitenedStepNorm(diff)` (the square-root-invariant Mahalanobis norm
  for CMA-ES; a covariance-free host returns ≤ 0 → the driver injects unclipped, and reports
  `HasProposalCovariance = false` → GN-only infill, no CMIX). DE/MOO hosts (Plan 05) implement this.
- **`Samplers/Janus/CmaPopulationHost.cs`** — the CMA-ES host: exactly the CMA-ES driving Phase 1
  entangled (initial mean/step in the transformed box, native resample-repaired `Ask`, coherent
  `Tell`, mean/σ/trace(C)/Mahalanobis whitening), moved verbatim behind the interface.
- **`Samplers/Janus/JanusOptimizer.cs`** — the driver now holds an `IPopulationHost` (not a `Cma`);
  the constructor takes the pre-built host + a CMIX-noise seed, reads `popsize` from the host, and its
  three covariance touch-points route through the interface. CMIX is gated on `HasProposalCovariance`.
- **`JanusSampler`** builds a `CmaPopulationHost` then the driver — the RNG seeding (host stream =
  `seed`, CMIX stream = `seed ^ 0x5A17`) and `popsize` are byte-identical to Phase 1, so the
  CMA-ES-hosted candidate/elite stream is unchanged.

**Naming note.** The plan called the driver `JanusInfill`; it kept the Phase-1 name `JanusOptimizer`
(class + file) because a concurrent session was actively editing that exact file (Harvest / MOO
component-loss prep) and a rename would have broken its pending edits. The name is cosmetic — the
host-agnostic seam is what Plan 05 depends on; a rename can follow once that work settles.

**Validation.** Phase-1 `JanusSamplerTests` (10) pass unchanged on both TFMs — including the
same-seed determinism and matches-or-beats-CMA-ES gates that would catch any stream divergence — and
`JanusMetricParityTests` still compiles against `JanusOptimizer.FitComponentJacobian` (unchanged).
New `JanusInfillHostTests` (2) drive the bare driver against a recording mock host, proving coherent
batching (popsize-sized generations, ask-order, partial/gapped generations withheld), native slot
fill, slot-0 elite reservation, and CMIX gated off for a covariance-free host.

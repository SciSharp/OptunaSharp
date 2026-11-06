# OptunaSharp — JANUS-adjacent porting plans

> **Status: DESIGN (not started).** These plans grew out of the 2026-09-10 JANUS port (arXiv:2608.22862;
> reference [github.com/hongyuanyu/JANUS](https://github.com/hongyuanyu/JANUS) @ `53e372f`, Apache-2.0). JANUS Phase 1 (the single-objective CMA-ES–hosted sampler)
> is implemented and byte-parity-tested; see the [`janus-port`](../../) memory and
> `src/OptunaSharp.Core/Samplers/Janus/`. Each plan here is either **required to finish JANUS** or
> **unlocked / made newly worthwhile by it**, and every one obeys the governing
> [`docs/PORTING_PLAN.md`](../PORTING_PLAN.md) tier rules (Core stays pure-managed — BCL + NumSharp only;
> anything needing a foreign runtime becomes a satellite package; algorithms with no Optuna counterpart
> are **beyond-core, validated by convergence** like `LmCmaSampler`).

These files are **ordered by dependency** — a plan's hard prerequisites always appear earlier. The
numeric prefix is the recommended execution order.

## Dependency graph

```
01 BBOB/COCO suite  ─────────────────────────────────┐  (validation substrate: reproduces the JANUS
                                                      │   paper tables and powers VolleyEval for EVERY
                                                      │   sampler — CMA, LM-CMA, TPE, GP, DE, JANUS)
02 component_losses channel ──┐                       │
03 JANUS infill-wrapper infra ┤ (needs JANUS Phase 1) │
04 SMS-EMOA ──────────────────┤ (needs HV infra ✓)    │
   AGE-MOEA-2  [in flight] ────┤                       │
   DE host     [built in 05] ──┤                       ▼
                              ▼                  (validation)
                     05 JANUS completion (Phase 2 DE + Phase 3 MOO)  ◄── the capstone

06 VkD-CMA          ── independent enrichment ── validated via 01 (reproduces Prop 1 / Table 9)
07 MOEA/D · RVEA · PSO ─ independent enrichment ─ validated via 01 (+ DTLZ/WFG for the MOEAs)
```

## Order & rationale

| # | Plan | Hard deps | Tier | Value / effort |
|---|------|-----------|------|----------------|
| [01](01-bbob-coco-benchmark-suite.md) | BBOB/COCO benchmark suite (f1–f24) | — | N | ★★★ / low |
| [02](02-component-losses-channel.md) | `component_losses` observed-components channel | — | N | ★ / low |
| [03](03-janus-infill-wrapper-infra.md) | JANUS infill-wrapper reusable infrastructure | JANUS Ph.1 (done) | N | ★★ / medium |
| [04](04-sms-emoa-sampler.md) | SMS-EMOA sampler | HV infra (present) | N | ★★ / medium |
| [05](05-janus-completion-de-moo-hosts.md) | **JANUS completion — all phases (DE host + MOO)** | 02, 03, 04, AGE-MOEA-2 (in flight) | N | ★★★ / high |
| [06](06-vkd-cma-sampler.md) | VkD-CMA sampler | Core CMA infra | N | ★★ / med-high |
| [07](07-moead-rvea-pso-baselines.md) | MOEA/D · RVEA · PSO baselines | Core | N | ★ / medium |
| [08](08-pymoo-interop-sampler-bridge.md) | pymoo sampler bridge (`Interop.Pymoo`) | pythonnet interop | **X** | ★★★ / med-high |
| [09](09-pymoo-live-parity-gate.md) | Live pymoo parity gate (AGE-MOEA-2) | AGE-MOEA-2 (done) | **X** | ★★ / low |

**Tier-X pymoo-interop workstream (08–09).** Separate from the Tier-N native ports above: unlocked by the
2026-09-10 pymoo-on-numpy-2.4.2 migration (pymoo now runs in the interop interpreter, commit `9c7aec2`),
these ship a `OptunaSharp.Interop.Pymoo` satellite that drives the **real** pymoo via pythonnet (the
BoTorch/PyCma pattern). 08 exposes pymoo's MOO algorithm zoo as `PymooSampler` (the live-runtime
complement to 07's selective native reimplementations); 09 turns the AGE-MOEA-2 differential corpus into a
standing **live** parity gate. Independent of the JANUS chain; 08 blocks nothing but enriches VolleyEval
with the literature's reference algorithms.

**Why this order.** 01 and 02 have no prerequisites and are foundational — 01 is the substrate that
lets us *quantify* every subsequent sampler (and validate JANUS Phase 1 against the paper), and 02 is
the interface JANUS's component mode and the MOO metric both consume. 03 generalizes the Phase-1
CMA-ES infill into a host-agnostic layer; 04 adds the hypervolume MOO host JANUS-SMS needs. 05 is the
JANUS capstone and depends on 02+03+04 (plus the in-flight AGE-MOEA-2 and a DE host it builds first).
06 and 07 are independent enrichments — they block nothing, but their *value* (reproducing Table 9,
populating MOO leaderboards) is realized through 01, so they follow it.

**Concurrency note.** LM-CMA and AGE-MOEA-2 are being ported in parallel sessions; 05's Phase 3 and 06
should dovetail with them (AGE-MOEA-2 is the JANUS-AGE2 host; VkD-CMA sits beside LM-CMA in the
restricted-covariance family). Re-check `git`/HEAD before starting any of these.

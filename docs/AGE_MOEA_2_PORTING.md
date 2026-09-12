# AGE-MOEA-2 — byte-perfect porting plan

> **Paper:** Annibale Panichella, *An Improved Pareto Front Modeling Algorithm for Large-scale
> Many-Objective Optimization*, GECCO 2022, doi:10.1145/3512290.3528732 (**AGE-MOEA-2**), building
> on Panichella, *An adaptive evolutionary algorithm based on non-Euclidean geometry for
> many-objective optimization*, GECCO 2019 (**AGE-MOEA**).
>
> **Why we care:** it is the multi-objective optimizer inside **Mimicry** (Weißl et al., *Targeted
> Deep Learning System Boundary Testing*, ACM TOSEM 2025, arXiv 2408.06258). It is **absent from
> Optuna core** (which stops at NSGA-II / NSGA-III / MO-TPE), so — like LM-CMA and JANUS — it is a
> *beyond-core, convergence-validated* sampler validated against its **authoritative reference
> implementation**, not against Optuna.

## 0. Verdict up front

Byte-perfection is **achievable**, but only against a **precisely pinned oracle**, and with **one
component demoted to a tolerance tier**:

| | Verdict |
|---|---|
| **RNG stream** (operators) | ✅ **Bit-exact** — pymoo 0.6.1.3 uses global **MT19937**, already ported as `LazyRandomState`. |
| **GA loop / operators / non-dominated sort / normalization** | ✅ **Bit-exact** — deterministic given the RNG + OpenBLAS linalg. |
| **Survival geometry kernels** (`find_zero`, `project_on_manifold`, geodesic distances) | ✅ **Bit-exact against JIT-disabled pymoo; empirically bit-exact vs stock fastmath too** — see §1.1. |
| **Selection decisions** (who survives, tournament winners, argmin/argmax picks) | ✅ **Identical** on the validation suite. |

"Byte-perfect with the official implementation" therefore means: **bit-identical to pymoo 0.6.1.3
executed with `NUMBA_DISABLE_JIT=1`** — and, as measured (§1.1), that path is itself **bit-identical
to stock (fastmath-JIT) pymoo** at the tested scales, so the guarantee extends to real Mimicry. The
`MisalignedRegistry` ledger guards the (so-far-unobserved) case where a larger/​pathological front
makes fastmath diverge.

> **Empirical update (measured 2026-09-10, isolated pymoo 0.6.1.3 + numba 0.60.0 + numpy 2.0.2):**
> fastmath-JIT vs `NUMBA_DISABLE_JIT=1` were **0-ULP identical in X and F with 0 rank flips** over
> full runs of ZDT1 (2-obj, pop 20, 10 gen), a Mimicry-like 2-obj box problem (n_var 8, pop 20, 15
> gen), and DTLZ2 (3-obj, pop 40, 20 gen). The "one real blocker" is, at these scales, **not a
> blocker** — the geometry kernels reduce over only 2–3 objectives (no reassociation freedom) and X/F
> are produced by the fastmath-free operators. True byte-exactness of the whole run is the target;
> close-parity is the fallback only if a future config trips the ledger.

---

## 1. The oracle — pin it exactly

Mimicry's `defaults/mimicry/environment.yml` pins **`pymoo==0.6.1.3`**, **`numba==0.60.0`**. Nothing
below is valid against any other pymoo — the RNG and operator internals changed materially after
0.6.1.x (see §2).

Reference source (pin every file to the `0.6.1.3` git tag):

- `pymoo/algorithms/moo/age2.py` — `AGEMOEA2`, `AGEMOEA2Survival`, `find_zero`, `project_on_manifold`.
- `pymoo/algorithms/moo/age.py` — `AGEMOEASurvival` (the base that AGEMOEA2 **inherits**),
  `survival_score`, `compute_geometry`, `find_corner_solutions`, `point_2_line_distance`, `normalize`.
- `pymoo/algorithms/moo/nsga2.py` — `binary_tournament`.
- `pymoo/operators/{crossover/sbx.py, mutation/pm.py, selection/tournament.py, sampling/rnd.py}`.
- `pymoo/core/{algorithm.py, genetic.py, duplicate.py, population.py, individual.py}`.
- `pymoo/util/nds/non_dominated_sorting.py`.

### 1.1 The fastmath problem and its resolution

In 0.6.1.3, `project_on_manifold` and `pairwise_distances` are `@jit(fastmath=True)` and `find_zero`
runs Newton's method over `log`/`exp`. `fastmath=True` lets LLVM reassociate, contract (FMA) and
flush-to-zero, so **even pymoo's own output is not portable across CPU/LLVM builds** at the ULP
level. We cannot chase a moving target.

**Resolution — a two-part oracle:**

1. **Bit-exact target =** pymoo 0.6.1.3 run with **`NUMBA_DISABLE_JIT=1`**. Then those kernels
   execute as **plain CPython/NumPy** under strict IEEE-754 + the platform libm. This repo already
   matches NumPy `Log`/`Exp`/`Pow` **bit-for-bit** (see the *numeric bit-exact parity* notes) and
   `np.linalg.solve` byte-for-byte via OpenBLAS, so a strict-IEEE C# port can hit it exactly.
2. **Decision-equivalence check =** run stock (fastmath-JIT) pymoo alongside and assert the ULP gap
   in the survival scores **never flips a selection decision** (survivor set, tournament winner,
   `argmin`/`argpartition`/`argmax` index) on the whole validation suite. Record the max observed
   ULP gap in a `MisalignedRegistry`-style ledger.

This makes the port **bit-identical to a reproducible reference execution** and **behaviorally
identical to real Mimicry**.

---

## 2. RNG feasibility — the make-or-break, and it breaks our way

`pymoo/core/algorithm.py @0.6.1.3` seeds like this:

```python
import random
random.seed(seed)          # Python stdlib Mersenne Twister
np.random.seed(seed)       # NumPy *legacy* global RandomState == MT19937
```

and every operator draws from the **global legacy stream** — e.g. `sbx.py`:
`cross = np.random.random((n_matings, n_var)) < prob_var`, `rand = np.random.random(len(eta))`, …

Consequences:

- **`np.random.*` legacy == MT19937 == `LazyRandomState`.** OptunaSharp already reproduces this exact
  generator bit-for-bit (it is the whole basis of Optuna parity). **Reuse it directly.** No PCG64
  port is needed for this pin. (For contrast: pymoo `main` switched to `np.random.default_rng` =
  **PCG64**; NumSharp 0.70.0 *does* expose PCG64/Generator, but we do **not** target that version.)
  **✅ Verified 2026-09-10:** `np.random.seed(S)` + `np.random.random(n)` / `random((r,c))` /
  `randint(0,10)` are **0-ULP identical** between numpy 2.0.2 and NumSharp 0.70.0 `np.random`
  (seeds 42 & 7), including 2-D **C-order** fill — so the operator stream ports bit-for-bit.
- **✅ Verified: stdlib `random` is seeded but never drawn from** on the AGEMOEA2 path (only
  `random.seed(seed)` in `core/algorithm.py`; every draw is `np.random.*`). No second RNG port needed.
- **Draw order and array shape are load-bearing.** `np.random.random((n_matings, n_var))` fills
  **C-order (row-major)**. The port must request the identical shape and consume the stream in the
  identical order per operator call. A transposed or per-gene loop that draws in a different order
  is a silent bit-break.
- **⚠️ stdlib `random.seed(seed)` caveat (verify first).** If *any* code on the AGEMOEA2 path draws
  from Python's stdlib `random` (a different MT19937 — different seeding + tempering than NumPy's),
  we would additionally need a CPython-`random` port. **Task 0** below greps the pinned pymoo path;
  the expectation is that the numeric operators use only `np.random`, and stdlib `random` is seeded
  but unused on this path.

---

## 3. What "port AGE-MOEA-2" actually includes

`AGEMOEA2Survival(AGEMOEASurvival)` overrides **only** `compute_geometry` and `pairwise_distances`;
everything else (normalization, corner solutions, the greedy survival loop, environmental selection)
is **inherited from AGE-MOEA (2019)**. So the deliverable is:

**AGE-MOEA (2019) survival base + the two AGE-MOEA-2 overrides + the pymoo GA scaffold + pymoo-faithful
operators + the GA driver loop.** A welcome side effect: **AGE-MOEA (2019) falls out for free** (same
base, default `compute_geometry`/`pairwise_distances`) as a second validated sampler.

### 3.1 Operators must be **pymoo-faithful, not Optuna's**

OptunaSharp already has SBX/PM/tournament under `Samplers/Nsgaii/` — **do not reuse them.** They are
byte-exact with *Optuna's* NSGA-II, whose SBX/PM draw randoms in a **different pattern** (different
per-gene loop, different `eta` handling, different bounded-value reflection) than pymoo's
`cross_sbx` / `mut_pm`. Byte-exactness with pymoo requires **pymoo's** operator code, ported draw-for-draw.

**Subtle RNG coupling — `eliminate_duplicates=True`.** `AGEMOEA2` enables pymoo's
`DefaultDuplicateElimination`: offspring duplicating the population are **re-mated**, consuming
**additional** RNG draws in a data-dependent amount. Skipping or reordering this desynchronizes the
entire downstream stream. Port `core/duplicate.py` + the mating retry loop **exactly**.

### 3.2 Placement in the repo

- **Runtime: pure-managed, no foreign runtime** → lives in **Core** (`Samplers/AgeMoea/`), same tier
  as `Nsgaii`/`Nsgaiii`/LM-CMA/JANUS. pymoo is a **test-time oracle only** (via pythonnet), never a
  runtime dependency.
- **Two layers, two strengths of guarantee:**
  - **Layer 1 — faithful GA core** (`AgeMoea2Optimizer`): a standalone `minimize()`-shaped loop that
    is **bit-exact vs `pymoo.optimize.minimize`**. This is where "byte-perfect" is proven.
  - **Layer 2 — `AgeMoea2Sampler : BaseSampler`** (the Optuna-idiomatic ask/tell face, mirroring
    `NSGAIISampler`, reconstructing the population from completed trials). Byte-exact on the operator
    draws; **behaviorally** validated end-to-end. Note explicitly: Optuna has no AGE-MOEA-2, so Layer 2
    has **no Optuna oracle** — its correctness is defined by Layer 1 + the pymoo semantics, not by an
    Optuna reference. Ship Layer 1's guarantee as the headline; Layer 2 is ergonomics.

---

## 4. Component inventory

| # | Component | pymoo source (`@0.6.1.3`) | Fidelity | Reuse / New |
|---|---|---|---|---|
| 1 | MT19937 + `random()`/`randint()`/`permutation` | `np.random` legacy | bit-exact | **Reuse** `LazyRandomState` |
| 2 | `FloatRandomSampling` | `sampling/rnd.py` | bit-exact | New (pymoo-faithful) |
| 3 | SBX crossover `cross_sbx` | `crossover/sbx.py` | bit-exact | New (≠ Optuna SBX) |
| 4 | PM mutation `mut_pm` + `mut_binomial` | `mutation/pm.py` | bit-exact | New (≠ Optuna PM) |
| 5 | `TournamentSelection` + `binary_tournament` (`comp_by_rank_and_crowding`) | `selection/tournament.py`, `nsga2.py` | bit-exact | New |
| 6 | Duplicate elimination + mating retry | `core/duplicate.py`, `core/mating.py` | bit-exact (RNG-coupled) | New |
| 7 | Non-dominated sorting | `util/nds/non_dominated_sorting.py` | bit-exact (deterministic) | New (or verify vs `Nsgaii` NDS) |
| 8 | Ideal point, `find_corner_solutions`, `point_2_line_distance`, `normalize` (intercept solve) | `age.py` | bit-exact (OpenBLAS `solve`) | **Reuse** `Internal/LinAlg`; New glue |
| 9 | `compute_geometry` + `find_zero` (Newton, log-space) | `age2.py` | bit-exact vs JIT-off / ULP vs fastmath | New |
| 10 | `project_on_manifold` + geodesic `pairwise_distances` | `age2.py` | bit-exact vs JIT-off / ULP vs fastmath | New |
| 11 | `survival_score` greedy maximin loop (`argpartition` 2-NN) | `age.py` | bit-exact (see ties, §6) | New |
| 12 | GA driver (`GeneticAlgorithm.advance`, `advance_after_initial_infill=True`, `n_offsprings=pop_size`) | `core/genetic.py`, `algorithm.py` | bit-exact | New (Layer 1) |
| 13 | `AgeMoea2Sampler : BaseSampler` ask/tell | — (mirror `NSGAIISampler`) | behavioral | New (Layer 2) |

---

## 5. Algorithm spec (implementer's reference)

Skeleton = NSGA-II (μ+λ): combine parents+offspring → non-dominated sort → fill by front → the one
splitting front is ranked by the **survival score**. Defaults for AGEMOEA2:
`pop_size=100, SBX(prob=0.9, eta=15), PM(eta=20), binary_tournament(comp_by_rank_and_crowding),
eliminate_duplicates=True`.

**Survival (per generation, on the first front F₀, then remaining fronts):**

1. **Ideal point** `z* = min(F₀, axis=0)`; shift `F₀ -= z*`.
2. **Corner/extreme solutions** — per objective, the point minimizing `point_2_line_distance` to that
   axis. Mark extremes with score `∞`, pre-select them.
3. **Normalize** — intercepts from solving the hyperplane through the extreme points
   (`np.linalg.solve`; fall back to per-axis max if singular/NaN/Inf); divide `F₀` by intercepts
   (zeros → 1). ← OpenBLAS, bit-exact.
4. **Geometry exponent `p`** — **AGE-MOEA-2:** pick the point with **smallest Euclidean norm** (most
   interior), then `p = find_zero(point, n, 1e-3)` solving `Σ xᵢ^p = 1` by **Newton in log-space**
   (regularized `+1e-10`, overflow-guarded, ≤100 iters, stop at `|Δx| ≤ 1e-3`). Clamp: `p ≤ 0.1 or
   NaN → 1.0`; `p > 20 → 20`. *(AGE-MOEA-1 instead uses the closed form `p = log(n)/log(1/mean(coords))`
   on the diagonal-nearest point.)*
5. **Pairwise distances (diversity metric) — AGE-MOEA-2:** `project_on_manifold(x,p) = x / ‖x⁺‖_p`
   (positive components only). Then if `0.95 < p < 1.05` → plain Euclidean between projected points;
   else **2-segment geodesic** `d(A,B) = ‖A−mid‖₂ + ‖mid−B‖₂` where `mid = project((A+B)/2, p)`.
   *(AGE-MOEA-1 uses raw Minkowski-`p` distances.)*
6. **Survival score (greedy maximin)** — start from the pre-selected extremes; repeatedly add the
   unselected point maximizing the **sum of distances to its two nearest already-selected neighbors**
   (`argpartition(..., 2)`), distances pre-divided by each point's `‖·‖_p` norm; record that sum as
   its score.
7. **Remaining fronts** `i ≥ 1`: score `= 1 / minkowski_p(point, ideal)` (proximity only).
8. **Environmental selection** — fill by front rank; within the splitting front keep highest scores.

`binary_tournament(comp_by_rank_and_crowding)`: lower front rank wins; tie → **higher survival
score** wins; tie → the pymoo tie-break (first index / random per `binary_tournament`), ported exactly.

---

## 6. Byte-exactness hazards & mitigations

| Hazard | Risk | Mitigation |
|---|---|---|
| `@jit(fastmath=True)` reassociation | Non-portable ULPs in kernels | Oracle = `NUMBA_DISABLE_JIT=1`; ledger the fastmath ULP gap + prove decision-preserving (§1.1) |
| `log`/`exp` in `find_zero` | ucrt/CoreCLR vs libm divergence | Repo matches NumPy Log/Exp bit-for-bit; Newton stop is coarse (`1e-3`, 100 iters) so sub-ULP noise can't change the iterate count — **verify iterate counts match** |
| RNG **draw order / ndarray shape** | Silent desync | Mirror every `np.random.random(shape)` call, C-order fill, exact call sequence per operator |
| `eliminate_duplicates` re-mating | Data-dependent extra draws | Port `duplicate.py` + retry loop exactly (§3.1) |
| stdlib `random` on the path | Would need a 2nd MT port | **Task 0**: grep the pinned path; expected unused |
| `np.argpartition` 2-NN ties | Introselect unstable on exact ties | `eliminate_duplicates=True` removes exact dupes; ties are measure-zero on continuous data — assert "no tie" on suite, else port introselect exactly |
| `np.argmin/argmax` | First-occurrence semantics | Match "first index of extremum" |
| `np.round(front, 12)` in `survival_score` | Rounding mode | Reproduce NumPy round-half-to-even at 12 dp |
| `np.linalg.solve` intercepts | Backend divergence | OpenBLAS in Core is byte-exact vs NumPy (verified) |
| Pairwise/`sum` order | Pairwise vs sequential | n = #objectives is tiny → sequential == pairwise; still match reduction order |

---

## 7. Validation strategy

Mirrors the repo's dual gate (committed differential oracle + live pythonnet interop), per the
`testing` skill and the LM-CMA / JANUS precedent.

- **Differential corpus oracle (no Python at test time).** A `test/oracle/gen_agemoea2_oracle.py`
  runs pymoo 0.6.1.3 **with `NUMBA_DISABLE_JIT=1`** over a grid and dumps **generation-by-generation**
  JSONL: seed, and for every generation the full `X`, `F`, ranks, survival scores, and survivor
  indices. C# **replays** the same seeds and asserts **bit-equality** at every generation (0-ULP on
  ranks/decisions; ≤ ledgered-ULP on kernel scores).
  - **Problems:** `ZDT1–3` (2-obj, convex/concave/disconnected), `DTLZ1–2` (3-obj, linear/spherical
    — exercises `p≈1` vs `p≈2` branches), plus **Mimicry's regime** (2 objectives, `n_var∈{8,14}`,
    box `[0,1]`, synthetic confidence-balance + sparsity objective). The 2-objective cases are the
    *primary* bit-exact target; many-objective DTLZ is the geometry *stress* target.
  - **Grid:** `pop_size∈{20,100}`, `n_gen∈{10,50}`, `seed∈{0,1,7,42,2024}`.
- **Live interop parity gate ✅ DONE** (`test/OptunaSharp.Tests.Interop/AgeMoea2ParityTests.cs`,
  `[TestCategory("RequiresPython")]`; plan [09](plans/09-pymoo-live-parity-gate.md)): drives **live**
  stock (JIT-on) pymoo 0.6.1.3 via pythonnet (like the BoTorch/PyCma bridges drive their references) and
  asserts the native `AgeMoea2Optimizer` matches **generation-by-generation bit-exact** over the corpus
  grid plus a larger 3-obj front, with the max-ULP decision-equivalence ledger folded into the final
  assertion (green ⇒ 0 ULP). Operator spot-checks (SBX/PM/tournament/sampling) run the corpus sections
  live to localize any drift. 21 cases green on net8.0 + net10.0; a +1-ULP smoke perturbation confirmed
  the gate bites. Host-pinned to x64 (Inconclusive on arm64); hard-fails under
  `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1` in CI (which now installs pymoo). *(JIT-off, the nominal
  bit-exact target of §1.1, is unavailable — `find_corner_solutions` uses `numba.intp` as a numpy dtype —
  and, as measured, identical to JIT-on at these scales anyway, so JIT-on is the live oracle.)*
- **Operator unit tests:** SBX/PM/tournament/sampling each replayed against pymoo in isolation
  (fixed seed, fixed parents) — bit-exact — so a desync is localized before the full loop.

---

## 8. Phasing (gate after each; pause for review — per the JANUS convention)

- **Task 0 — Recon (½ day).** Grep the pinned `pymoo==0.6.1.3` path for stdlib `random` usage;
  confirm MT19937-only. Stand up `gen_agemoea2_oracle.py` (JIT-disabled) + one ZDT1 dump.
- **Phase 1 — Operators bit-exact (core RNG coupling).** `FloatRandomSampling`, `cross_sbx`,
  `mut_pm`+`mut_binomial`, `TournamentSelection`, duplicate-elimination retry. Gate: each operator
  0-ULP vs pymoo on fixed seeds/inputs. *This is the highest-risk phase — it pins the RNG stream.*
- **Phase 2 — Non-dominated sort + normalization.** NDS ranks + `find_corner_solutions` +
  `normalize` (OpenBLAS solve). Gate: 0-ULP ranks + normalized fronts on the corpus.
- **Phase 3 — Survival geometry (AGE-MOEA base + AGE-MOEA-2 overrides).** `find_zero`,
  `project_on_manifold`, geodesic `pairwise_distances`, greedy `survival_score`. Gate: bit-exact vs
  JIT-off; ULP ledger vs fastmath; **zero decision flips**. (Ship AGE-MOEA-1 here too — same base.)
- **Phase 4 — GA driver (Layer 1).** Full `minimize`-shaped loop, `advance_after_initial_infill`,
  offspring count, termination. Gate: **generation-by-generation bit-exact vs `pymoo.minimize`** on
  the whole grid; commit the corpus oracle.
- **Phase 5 — Layer 2 sampler + interop gate. ✅ live gate DONE.** `AgeMoea2Sampler : BaseSampler`
  (ask/tell, population-from-trials like `NSGAIISampler`); **live pythonnet gate green both TFMs**
  (`AgeMoea2ParityTests`, plan [09](plans/09-pymoo-live-parity-gate.md)); DocFX note; optional
  `examples/` showcase.

---

## 9. Effort, risks, DoD

**Effort:** Layer 1 core ≈ 1.0–1.5k LOC (operators ~400, survival ~450, NDS ~200, driver+dup ~300);
Layer 2 adapter ~300; oracle harness + tests ~600. **≈ 2–2.5k LOC incl. tests.** Calendar: ~2–3
focused weeks to full interop-green, dominated by chasing bit-exact operators (Phase 1) and
characterizing the fastmath gap (Phase 3) — **not** the algorithm, which is small and well-specified.

**Top risks:** (R1) stdlib `random` on the path → a second MT port [mitigate: Task 0 first].
(R2) duplicate-elimination RNG desync [Phase 1 isolates it]. (R3) `argpartition` ties [assert-no-tie;
else port introselect]. (R4) fastmath gap flips a decision on some exotic geometry → that kernel
stays *tolerance* not *bit-exact* (acceptable, decision-preserving, ledgered). (R5) Layer 2 cannot be
simultaneously pymoo-bit-exact and Optuna-idiomatic → **scope the bit-exact guarantee to Layer 1**.

**Definition of Done:** generation-by-generation **bit-identical to pymoo 0.6.1.3 (JIT-disabled)** and
**decision-identical to stock pymoo** across the grid; committed corpus oracle replays with **no
Python**; live interop gate green on net8+net10; AGE-MOEA-1 validated as the by-product; `LazyRandomState`
(never `System.Random`); Core stays pure-managed (pymoo is test-time only); any residual kernel
tolerance recorded in the `MisalignedRegistry` ledger.

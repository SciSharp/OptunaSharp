# Particle Swarm Optimization (1995)

> 🚧 **Stub** — a full write-up (idea, pseudocode, OptunaSharp usage) is coming. Back to the [Methods overview](index.md).

**Paper:** [Kennedy & Eberhart, Proc. IEEE ICNN 1995](https://doi.org/10.1109/ICNN.1995.488968)


## OptunaHub's `PSOSampler` is a different sampler

OptunaSharp's `PsoSampler` follows Kennedy & Eberhart with Shi & Eberhart's linearly decreasing inertia. OptunaHub
also has a PSO ([`samplers/pso`](https://hub.optuna.org/samplers/pso/), `PSOSampler`, since 2025-09), with a
constant inertia, and OptunaSharp does not follow it. A study moved over from the hub package therefore gets other
defaults and other trials, even with the same seed:

| OptunaHub `PSOSampler` | OptunaSharp `PsoSampler` |
|---|---|
| `n_particles=10` | `popsize`: `max(10, 4 + ⌊3 ln d⌋)` |
| `inertia=0.5`, constant | `w = 0.9`, decreasing linearly to `wMin = 0.4` over `inertiaDecayGenerations = 100` |
| `cognitive=1.5`, `social=1.5` | `c1 = c2 = 1.49445` |
| velocity clamped to the full range of each parameter | `maxVelocityRate = 0.2` of the unit box |
| `search_space`, `seed` | `seed`; the search space is inferred; plus `nStartupTrials`, `independentSampler`, `warnIndependentSampling` |
